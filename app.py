# ==============================================================
# IMPORTS
# ==============================================================
import pandas as pd
import plotly.express as px
import streamlit as st


# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="Customer Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
)


# ==============================================================
# DATA LOADING & CLEANING
# ==============================================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    # Standardise sentiment labels
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.title()

    # Remove invalid rows for analysis stability
    df = df.dropna(subset=["country", "timestamp"])

    # Convert timestamp safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])

    # Ensure text column fallback exists safely
    if "processed_review" in df.columns and "_clean_text" in df.columns:
        df["processed_review"] = df["processed_review"].fillna(df["_clean_text"])

    return df


df = load_data("processed_sentiment_data.csv")


# ==============================================================
# HEADER
# ==============================================================
st.markdown("## 📊 Customer Sentiment Dashboard")
st.caption("Analyse customer feedback across countries and product categories")


# ==============================================================
# DATE FILTER
# ==============================================================
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

date_range = st.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

start_date, end_date = date_range if isinstance(date_range, tuple) else (min_date, max_date)


# ==============================================================
# SIDEBAR FILTERS
# ==============================================================
st.sidebar.header("Filters")

countries = st.sidebar.multiselect(
    "Country",
    sorted(df["country"].dropna().unique()),
    default=sorted(df["country"].dropna().unique()),
)

categories = st.sidebar.multiselect(
    "Product Category",
    sorted(df["product_category"].dropna().unique()),
    default=sorted(df["product_category"].dropna().unique()),
)


# ==============================================================
# FILTER DATASET
# ==============================================================
filtered_df = df[
    (df["country"].isin(countries)) &
    (df["product_category"].isin(categories)) &
    (df["timestamp"].dt.date >= start_date) &
    (df["timestamp"].dt.date <= end_date)
]

if filtered_df.empty:
    st.warning("No data for selected filters.")
    st.stop()


# ==============================================================
# KPI CALCULATIONS
# ==============================================================
total = len(filtered_df)

sent_counts = filtered_df["sentiment"].value_counts()

positive_pct = sent_counts.get("Positive", 0) / total * 100
neutral_pct = sent_counts.get("Neutral", 0) / total * 100
negative_pct = sent_counts.get("Negative", 0) / total * 100


# ==============================================================
# KPI CARDS
# ==============================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Reviews", f"{total:,}")
c2.metric("Positive %", f"{positive_pct:.1f}%")
c3.metric("Neutral %", f"{neutral_pct:.1f}%")
c4.metric("Negative %", f"{negative_pct:.1f}%")


# ==============================================================
# CATEGORY REVIEWS + SENTIMENT DOUGHNUT
# ==============================================================
category_df = (
    filtered_df.groupby(["product_category", "sentiment"])
    .size()
    .reset_index(name="Reviews")
)

category_order = (
    category_df.groupby("product_category")["Reviews"]
    .sum()
    .sort_values(ascending=True)
    .index
    .tolist()
)

sentiment_df = filtered_df["sentiment"].value_counts().reset_index()
sentiment_df.columns = ["Sentiment", "Count"]

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Reviews by Product Category")
    fig_category = px.bar(
        category_df,
        x="Reviews",
        y="product_category",
        color="sentiment",
        orientation="h",
        barmode="stack",
        category_orders={"product_category": category_order},
        labels={
            "product_category": "Product Category",
            "sentiment": "Sentiment",
        },
    )
    fig_category.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title=None,
        xaxis_title="Reviews",
        legend_title_text="Sentiment",
    )
    st.plotly_chart(fig_category, use_container_width=True)

with right_col:
    st.subheader("Overall Sentiment")
    fig_sentiment = px.pie(
        sentiment_df,
        names="Sentiment",
        values="Count",
        hole=0.6,
    )
    fig_sentiment.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=30, b=10),
        legend_title_text="Sentiment",
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)


# ==============================================================
# REVIEW TRENDS + TOP GOOD-REVIEW COUNTRIES
# ==============================================================
trend = (
    filtered_df.groupby([pd.Grouper(key="timestamp", freq="M"), "sentiment"])
    .size()
    .reset_index(name="Reviews")
)

country_df = (
    filtered_df.groupby(["country", "sentiment"])
    .size()
    .reset_index(name="Reviews")
)

good_country_df = (
    country_df[country_df["sentiment"] == "Positive"]
    .groupby("country", as_index=False)["Reviews"]
    .sum()
    .rename(columns={"Reviews": "Good Reviews"})
    .nlargest(8, "Good Reviews")
    .sort_values("Good Reviews", ascending=True)
)

trend_col, country_col = st.columns(2)

with trend_col:
    st.subheader("Review Trends")
    fig_trend = px.line(
        trend,
        x="timestamp",
        y="Reviews",
        color="sentiment",
        labels={
            "timestamp": "Month",
            "sentiment": "Sentiment",
        },
    )
    fig_trend.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=30, b=10),
        legend_title_text="Sentiment",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with country_col:
    st.subheader("Top 8 Countries with Good Reviews")
    if good_country_df.empty:
        st.info("No positive reviews are available for the selected filters.")
    else:
        fig_country = px.bar(
            good_country_df,
            x="Good Reviews",
            y="country",
            orientation="h",
            labels={"country": "Country"},
        )
        fig_country.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title=None,
            xaxis_title="Good Reviews",
            showlegend=False,
        )
        st.plotly_chart(fig_country, use_container_width=True)


# ==============================================================
# BUSINESS INSIGHTS
# ==============================================================
country_sentiment_pct = pd.crosstab(filtered_df["country"], filtered_df["sentiment"], normalize="index") * 100
positive_country_rate = country_sentiment_pct.get("Positive", pd.Series(dtype=float))
negative_country_rate = country_sentiment_pct.get("Negative", pd.Series(dtype=float))
best_country = positive_country_rate.idxmax() if not positive_country_rate.empty else "N/A"
worst_country = negative_country_rate.idxmax() if not negative_country_rate.empty else "N/A"

category_sentiment_pct = pd.crosstab(filtered_df["product_category"], filtered_df["sentiment"], normalize="index") * 100
positive_category_rate = category_sentiment_pct.get("Positive", pd.Series(dtype=float))
negative_category_rate = category_sentiment_pct.get("Negative", pd.Series(dtype=float))
best_category = positive_category_rate.idxmax() if not positive_category_rate.empty else "N/A"
worst_category = negative_category_rate.idxmax() if not negative_category_rate.empty else "N/A"

i1, i2, i3, i4 = st.columns(4)

i1.info(f"🏆 {best_country} strongest performance")
i2.warning(f"⚠ {worst_category} needs attention")
i3.success(f"⭐ {best_category} best category")
i4.info(f"📉 Improve experience in {worst_country}")


# ==============================================================
# DATA TABLE (FIXED)
# ==============================================================
with st.expander("View Dataset"):
    st.dataframe(
        filtered_df,
        use_container_width=True,   # ✅ FIXED (was width="stretch")
    )


# ==============================================================
# DOWNLOAD (FIXED ENCODING)
# ==============================================================
st.download_button(
    label="Download Filtered Data",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sentiment_data.csv",
    mime="text/csv",
)