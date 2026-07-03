"""
Deployable DistilBERT sentiment classifier.

Run as a Streamlit app:
    streamlit run "Deploying DistilBERT Model.py"

Run as a command-line batch scorer:
    python "Deploying DistilBERT Model.py" --input review_data.csv --output predictions.csv

The script expects a Hugging Face DistilBERT sequence-classification model in the
same folder by default: config.json, tokenizer.json, tokenizer_config.json, and
model.safetensors or pytorch_model.bin.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path(os.getenv("SENTIMENT_MODEL_DIR", APP_DIR)).resolve()
DEFAULT_MAX_LENGTH = int(os.getenv("SENTIMENT_MAX_LENGTH", "96"))
DEFAULT_BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "32"))

# The trained config in this project uses LABEL_0/LABEL_1/LABEL_2, so these
# human-readable labels are used whenever the config labels are generic.
DEFAULT_LABEL_BY_ID = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

TEXT_COLUMN_CANDIDATES = [
    "review",
    "processed_review",
    "_clean_text",
    "review_text",
    "text",
    "feedback",
    "comment",
    "message",
]


@dataclass(frozen=True)
class ModelBundle:
    """Container for loaded model resources used during prediction."""

    tokenizer: object
    model: object
    torch: object
    device: object
    label_by_id: dict[int, str]
    max_length: int


def import_streamlit():
    """Import Streamlit only when the app UI is being used."""
    try:
        import streamlit as st

        return st
    except ImportError:
        return None


def running_inside_streamlit() -> bool:
    """Return True when executed by `streamlit run` instead of plain Python."""
    # Avoid importing Streamlit in CLI mode because get_script_run_ctx can emit
    # noisy warnings when no Streamlit runtime exists.
    if "streamlit" not in sys.modules:
        return False

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def validate_model_dir(model_dir: Path) -> None:
    """Fail early with a clear message if deployment files are missing."""
    required = ["config.json", "tokenizer.json"]
    missing = [name for name in required if not (model_dir / name).exists()]

    has_weights = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
    if not has_weights:
        missing.append("model.safetensors or pytorch_model.bin")

    if missing:
        raise FileNotFoundError(
            f"Missing model deployment file(s) in {model_dir}: {', '.join(missing)}"
        )


def load_label_mapping(model_dir: Path) -> dict[int, str]:
    """Read id2label from config.json, falling back to known sentiment labels."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return DEFAULT_LABEL_BY_ID.copy()

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        raw_mapping = config.get("id2label", {})
        label_by_id = {int(idx): str(label).lower() for idx, label in raw_mapping.items()}
    except Exception:
        return DEFAULT_LABEL_BY_ID.copy()

    # Hugging Face defaults such as LABEL_0 are not meaningful to end users.
    if not label_by_id or all(label.startswith("label_") for label in label_by_id.values()):
        return DEFAULT_LABEL_BY_ID.copy()

    return label_by_id


def load_distilbert_model(
    model_dir: Path = DEFAULT_MODEL_DIR,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> ModelBundle:
    """Load tokenizer/model once and move the model to CPU or GPU."""
    try:
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "Missing deployment dependencies. Install them with: "
            "pip install pandas torch transformers safetensors streamlit"
        ) from exc

    model_dir = Path(model_dir).resolve()
    validate_model_dir(model_dir)

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        device=device,
        label_by_id=load_label_mapping(model_dir),
        max_length=max_length,
    )


def get_cached_model_loader(st):
    """Use Streamlit cache in the UI, while keeping CLI loading simple."""

    @st.cache_resource(show_spinner=False)
    def cached_loader(model_dir: str, max_length: int) -> ModelBundle:
        return load_distilbert_model(Path(model_dir), max_length=max_length)

    return cached_loader


def find_text_columns(df: pd.DataFrame) -> list[str]:
    """Rank likely text columns first, then include other object/string columns."""
    preferred = [col for col in TEXT_COLUMN_CANDIDATES if col in df.columns]
    other_text_columns = [
        col
        for col in df.columns
        if (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]))
        and col not in preferred
    ]
    return preferred + other_text_columns


def prepare_texts(values: Iterable[object]) -> pd.Series:
    """Convert uploaded values into clean non-null strings for prediction."""
    return pd.Series(values, dtype="object").fillna("").astype(str).str.strip()


def predict_sentiment(
    texts: Iterable[object],
    bundle: ModelBundle,
    batch_size: int = DEFAULT_BATCH_SIZE,
    return_probabilities: bool = False,
) -> pd.DataFrame:
    """Predict sentiment labels for a list/series of review texts."""
    clean_texts = prepare_texts(texts)
    results = pd.DataFrame(index=clean_texts.index)
    results["predicted_sentiment"] = "unknown"
    results["predicted_encoded"] = -1
    results["confidence"] = 0.0

    valid_mask = clean_texts.ne("")
    valid_texts = clean_texts[valid_mask].tolist()
    if not valid_texts:
        return results

    labels: list[str] = []
    encoded_ids: list[int] = []
    confidences: list[float] = []
    probability_rows: list[list[float]] = []

    for start in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[start : start + batch_size]
        encoded = bundle.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=bundle.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(bundle.device) for key, value in encoded.items()}

        with bundle.torch.no_grad():
            outputs = bundle.model(**encoded)
            probabilities = bundle.torch.softmax(outputs.logits, dim=1)
            confidence_values, predicted_ids = probabilities.max(dim=1)

        batch_ids = predicted_ids.cpu().tolist()
        batch_confidences = confidence_values.cpu().tolist()

        encoded_ids.extend(batch_ids)
        confidences.extend(batch_confidences)
        labels.extend(bundle.label_by_id.get(idx, f"label_{idx}") for idx in batch_ids)

        if return_probabilities:
            probability_rows.extend(probabilities.cpu().tolist())

    results.loc[valid_mask, "predicted_sentiment"] = labels
    results.loc[valid_mask, "predicted_encoded"] = encoded_ids
    results.loc[valid_mask, "confidence"] = confidences

    if return_probabilities and probability_rows:
        label_order = [bundle.label_by_id.get(i, f"label_{i}") for i in range(len(probability_rows[0]))]
        probability_df = pd.DataFrame(
            probability_rows,
            columns=[f"prob_{label}" for label in label_order],
            index=results.index[valid_mask],
        )
        results = results.join(probability_df)

    return results


def build_insight(sentiment_counts: pd.Series, total_rows: int) -> str:
    """Create a short business-readable summary from predicted sentiments."""
    if total_rows == 0 or sentiment_counts.empty:
        return "No valid reviews were available for prediction."

    dominant = str(sentiment_counts.idxmax())
    dominant_pct = sentiment_counts.max() / total_rows * 100
    negative_pct = sentiment_counts.get("negative", 0) / total_rows * 100

    if dominant == "negative":
        recommendation = "Prioritise complaint themes and service recovery actions."
    elif dominant == "positive":
        recommendation = "Use the strongest feedback themes to reinforce what customers value."
    elif dominant == "neutral":
        recommendation = "Review neutral comments for ways to turn adequate experiences into strong ones."
    else:
        recommendation = "Review rows marked as unknown before making decisions."

    return (
        f"The dominant sentiment is {dominant.title()} "
        f"({dominant_pct:.1f}% of analysed reviews). Negative feedback represents "
        f"{negative_pct:.1f}% of the upload. {recommendation}"
    )


def score_dataframe(
    df: pd.DataFrame,
    text_column: str | None,
    bundle: ModelBundle,
    batch_size: int,
    include_probabilities: bool,
) -> pd.DataFrame:
    """Validate the input DataFrame and append prediction columns."""
    if df.empty:
        raise ValueError("Input data is empty.")

    if text_column is None:
        candidates = find_text_columns(df)
        if not candidates:
            raise ValueError("No text columns found. Add a review/text/comment column.")
        text_column = candidates[0]

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' was not found. Available columns: {list(df.columns)}")

    prediction_df = predict_sentiment(
        df[text_column],
        bundle=bundle,
        batch_size=batch_size,
        return_probabilities=include_probabilities,
    )

    return pd.concat([df.reset_index(drop=True), prediction_df.reset_index(drop=True)], axis=1)


def render_streamlit_app() -> None:
    """Render the upload-and-predict Streamlit application."""
    st = import_streamlit()
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install it with: pip install streamlit")

    st.set_page_config(
        page_title="DistilBERT Sentiment Analysis",
        page_icon="📈",
        layout="wide",
    )

    st.title("DistilBERT Customer Feedback Sentiment Analysis")
    st.caption("Upload customer feedback and classify each review with the deployed DistilBERT model.")

    model_dir = Path(
        st.sidebar.text_input("Model directory", value=str(DEFAULT_MODEL_DIR))
    ).expanduser()
    batch_size = st.sidebar.slider(
        "Prediction batch size",
        min_value=8,
        max_value=128,
        value=DEFAULT_BATCH_SIZE,
        step=8,
    )
    max_length = st.sidebar.slider(
        "Maximum token length",
        min_value=32,
        max_value=256,
        value=DEFAULT_MAX_LENGTH,
        step=16,
    )
    include_probabilities = st.sidebar.checkbox("Include class probabilities", value=True)

    cached_loader = get_cached_model_loader(st)
    try:
        bundle = cached_loader(str(model_dir), max_length)
    except Exception as exc:
        st.error(f"Unable to load DistilBERT model: {exc}")
        st.stop()

    st.sidebar.success(f"Model loaded on {str(bundle.device).upper()}")

    tab_upload, tab_single = st.tabs(["CSV batch scoring", "Single review"])

    with tab_single:
        review_text = st.text_area("Review text", height=140, placeholder="Paste one customer review here...")
        if st.button("Predict Review", type="primary"):
            if not review_text.strip():
                st.warning("Enter review text before predicting.")
            else:
                prediction = predict_sentiment(
                    [review_text],
                    bundle=bundle,
                    batch_size=1,
                    return_probabilities=True,
                ).iloc[0]
                st.metric("Predicted sentiment", str(prediction["predicted_sentiment"]).title())
                st.metric("Confidence", f"{float(prediction['confidence']):.1%}")
                st.dataframe(prediction.to_frame("value"), use_container_width=True)

    with tab_upload:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is None:
            st.info("Upload a CSV file to begin sentiment prediction.")
            return

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to read uploaded CSV file: {exc}")
            st.stop()

        if df.empty:
            st.warning("The uploaded CSV file is empty.")
            st.stop()

        st.subheader("Uploaded Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        candidate_text_columns = find_text_columns(df)
        if not candidate_text_columns:
            st.error("No text columns were found in the uploaded file.")
            st.stop()

        text_column = st.selectbox("Review text column", candidate_text_columns)

        if st.button("Run Sentiment Prediction", type="primary"):
            with st.spinner("Running DistilBERT sentiment predictions..."):
                try:
                    results_df = score_dataframe(
                        df=df,
                        text_column=text_column,
                        bundle=bundle,
                        batch_size=batch_size,
                        include_probabilities=include_probabilities,
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.stop()

            st.subheader("Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            counts = results_df["predicted_sentiment"].value_counts().sort_index()
            chart_data = counts.rename_axis("sentiment").reset_index(name="count")

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows scored", f"{len(results_df):,}")
            col2.metric("Valid predictions", f"{(results_df['predicted_sentiment'] != 'unknown').sum():,}")
            col3.metric("Average confidence", f"{results_df['confidence'].mean():.1%}")

            st.subheader("Sentiment Distribution")
            st.bar_chart(chart_data, x="sentiment", y="count")

            st.subheader("Key Business Insight")
            st.write(build_insight(counts, len(results_df)))

            st.download_button(
                label="Download Prediction Results",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="distilbert_sentiment_results.csv",
                mime="text/csv",
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for batch or single-text scoring."""
    parser = argparse.ArgumentParser(description="Score review sentiment with a deployed DistilBERT model.")
    parser.add_argument("--input", type=Path, help="CSV file to score.")
    parser.add_argument("--output", type=Path, help="Where to save scored CSV output.")
    parser.add_argument("--text", type=str, help="Single review text to score.")
    parser.add_argument("--text-column", type=str, default=None, help="Text column in the input CSV.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory containing model files.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Prediction batch size.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum token length.")
    parser.add_argument("--include-probabilities", action="store_true", help="Add probability columns to output.")
    return parser.parse_args(argv)


def run_cli(argv: list[str]) -> int:
    """Command-line entry point for deployment smoke tests and batch jobs."""
    args = parse_args(argv)

    if not args.input and not args.text:
        print(
            "No CLI input supplied. For the web app, run: "
            'streamlit run "Deploying DistilBERT Model.py"',
            file=sys.stderr,
        )
        return 2

    try:
        bundle = load_distilbert_model(args.model_dir, max_length=args.max_length)

        if args.text:
            result = predict_sentiment(
                [args.text],
                bundle=bundle,
                batch_size=1,
                return_probabilities=True,
            )
            print(result.to_json(orient="records", indent=2))
            return 0

        df = pd.read_csv(args.input)
        results_df = score_dataframe(
            df=df,
            text_column=args.text_column,
            bundle=bundle,
            batch_size=args.batch_size,
            include_probabilities=args.include_probabilities,
        )

        output_path = args.output or args.input.with_name(f"{args.input.stem}_scored.csv")
        results_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved scored output to: {output_path}")
        return 0

    except Exception as exc:
        print(f"Deployment scoring failed: {exc}", file=sys.stderr)
        return 1


if running_inside_streamlit():
    render_streamlit_app()
elif __name__ == "__main__":
    raise SystemExit(run_cli(sys.argv[1:]))
