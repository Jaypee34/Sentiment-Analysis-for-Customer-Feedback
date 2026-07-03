# AMDARI – Sentiment Analysis for Customer Feedback

## Project Overview

AMDARI is an end-to-end **Natural Language Processing (NLP)** and **Machine Learning** project designed to analyse customer feedback and transform unstructured text data into actionable business intelligence. The system classifies customer reviews into **Positive, Negative, and Neutral** sentiment categories while identifying the underlying drivers influencing customer satisfaction and dissatisfaction.

The project combines **classical machine learning models**, **transformer-based deep learning models**, **Explainable Artificial Intelligence (XAI)** techniques, and **real-time deployment** through an interactive web application.


## Project Objectives

The primary objective of this project is to build an intelligent sentiment analysis system capable of:

* Automatically classifying customer feedback into sentiment categories
* Understanding the linguistic patterns driving customer sentiment
* Explaining model decisions using interpretable AI methods
* Generating business insights from customer feedback data
* Supporting data-driven business decision-making through deployment dashboards


## System Architecture

Customer Feedback Data
        │
        ▼
Data Preprocessing
(Text Cleaning, Tokenisation, Lemmatization)
        │
        ▼
Feature Engineering
(TF-IDF Vectorisation + DistilBERT Embeddings)
        │
        ▼
Model Training
(Logistic Regression | Naive Bayes | SVM | DistilBERT)
        │
        ▼
Model Evaluation
(Accuracy, F1 Score, Cross Validation, Confusion Matrix)
        │
        ▼
Explainability Layer
(LIME Feature Importance Analysis)
        │
        ▼
Business Insight Generation
(Customer Pain Points + Satisfaction Drivers)
        │
        ▼
Deployment Layer
(Streamlit Real-Time Sentiment Dashboard)


## Technologies Used

### Programming Language

* Python

### Data Processing Libraries

* Pandas
* NumPy

### Natural Language Processing

* NLTK
* Scikit-learn
* TF-IDF Vectorizer

### Deep Learning & Transformers

* PyTorch
* TensorFlow / Keras
* Hugging Face Transformers
* DistilBERT

### Explainable AI

* LIME (Local Interpretable Model-Agnostic Explanations)

### Deployment & Visualisation

* Streamlit
* Plotly
* Matplotlib

## Data Preprocessing Pipeline

The raw customer feedback data passes through a structured preprocessing pipeline before model training.

Preprocessing steps include:

* Removal of punctuation and special characters
* Text normalisation
* Tokenisation
* Stopword removal
* Lemmatization
* Text vectorisation using TF-IDF
* Contextual embedding generation using DistilBERT
* Dataset balancing for class distribution improvement

The preprocessing pipeline ensures high-quality input data for accurate model training.


## Machine Learning Models Implemented

The project compares multiple supervised learning models.

### Classical Machine Learning Models

* Logistic Regression
* Complement Naive Bayes
* Support Vector Machine (Linear SVM)

### Deep Learning Model

* DistilBERT Transformer Model

Each model was evaluated using:

* Accuracy Score
* Macro F1 Score
* Stratified Cross Validation
* Confusion Matrix
* Overfitting Gap Analysis

## Explainable AI Layer

To improve transparency and interpretability, the project integrates **LIME Explainability**.

LIME identifies the words that contribute most strongly to sentiment classification.

This helps explain:

### Negative Sentiment Drivers

Examples:

* delayed delivery
* refund issues
* damaged products
* poor customer service
* defective products

### Positive Sentiment Drivers

Examples:

* excellent quality
* fast delivery
* reliable service
* easy purchase process
* helpful support

### Neutral Sentiment Drivers

Examples:

* average experience
* standard delivery
* acceptable product quality
* routine customer interactions

This explainability layer makes the model more transparent and suitable for business decision-maki

## Business Insight Generation

The system automatically converts sentiment predictions into stakeholder-friendly business insights.

Examples include:

* Identification of product categories generating high negative sentiment
* Detection of countries with poor customer satisfaction levels
* Recognition of customer satisfaction drivers
* Automated recommendations for operational improvements

The system moves beyond prediction and supports strategic business intelligence.

## Deployment Application

The project includes a real-time deployment application built with Streamlit.

Application capabilities include:

* CSV file upload for bulk customer feedback analysis
* Real-time sentiment prediction
* Sentiment distribution visualisation
* Interactive dashboards
* Automated business insight generation
* Continuous monitoring interface for performance tracking


## Example Deployment Workflow

Upload Customer Feedback CSV
          │
          ▼
Automatic Text Processing
          │
          ▼
Sentiment Prediction Engine
          │
          ▼
Prediction Results Table
          │
          ▼
Sentiment Distribution Dashboard
          │
          ▼
Business Insight Generation
          │
          ▼
Stakeholder Decision Support


## Project Business Value

The system enables organisations to:

* Monitor customer satisfaction continuously
* Detect service failures early
* Identify operational weaknesses
* Improve customer experience strategies
* Support evidence-based decision-making through AI

## Future Improvements

Potential future improvements include:

* Real-time API deployment
* Cloud deployment using Docker and AWS
* Multilingual sentiment analysis
* Advanced drift detection monitoring
* Continuous retraining pipeline
* Integration with customer service platforms


## Author

Eugene Osae -  Data Scientist at AMDARI INC
