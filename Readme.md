# CandiSight – AI‑Powered Hiring & ATS System

## Project Overview

CandiSight is an AI-driven hiring assistance platform designed to help HR professionals and recruiters quickly evaluate job applicants. The system analyzes candidate resumes against job descriptions to predict whether a candidate is a good fit and generates an ATS-style compatibility score. By combining Natural Language Processing (NLP) and Machine Learning, CandiSight reduces manual resume screening time, improves consistency, and supports data-driven hiring decisions.

The project is built as a hackathon-ready Minimum Viable Product (MVP) with a clear end-to-end pipeline: text preprocessing, feature extraction, similarity analysis, machine learning classification, and result visualization through a user-friendly interface.

---

## Problem Statement

Recruiters often receive hundreds of resumes for a single job role. Manual screening is time-consuming, subjective, and error-prone. Traditional Applicant Tracking Systems (ATS) are often rigid and lack transparency.

CandiSight addresses this problem by:

* Automatically comparing resumes with job descriptions
* Predicting candidate-job fit (Yes/No)
* Providing an explainable ATS compatibility score

---

## Solution Description

CandiSight uses NLP techniques to convert unstructured resume and job description text into numerical features. A machine learning model then predicts candidate suitability, while a rule-based scoring mechanism computes an ATS score based on skill and keyword overlap.

The system is designed to be:

* Fast and lightweight
* Explainable to non-technical users
* Easy to integrate with web applications

---

## Key Features

* Resume and Job Description text analysis
* Job fit prediction using Machine Learning (classification)
* ATS compatibility score (0–100)
* Skill and keyword matching visualization
* Modular ML pipeline for easy extension

---

## What to Prepare (Hackathon Focus)

### 1. Data Preparation

* Resume text dataset (from Kaggle or public sources)
* Job description dataset
* Predefined skill dictionary (e.g., Python, ML, SQL, React)
* Synthetic label generation using skill overlap logic

### 2. Machine Learning Pipeline

* Text cleaning: lowercase, punctuation removal, stopword removal, lemmatization
* Feature extraction using TF-IDF Vectorization
* Similarity computation using Cosine Similarity
* Classification model: Logistic Regression (primary)
* Evaluation using F1-score and confusion matrix

### 3. ATS Scoring Logic

* Skill match percentage
* Keyword overlap percentage
* Experience alignment (basic heuristic)
* Weighted scoring to produce final ATS score

### 4. Model Deployment Assets

* Trained ML model saved as .pkl file
* TF-IDF vectorizer saved as .pkl file
* Single prediction function for backend integration

---

## System Architecture

1. User uploads resume and job description
2. Backend preprocesses text data
3. TF-IDF vectorizer transforms text into features
4. ML model predicts job fit
5. ATS score is calculated using rule-based logic
6. Results are displayed on the frontend

---

## Technology Stack

* Programming Language: Python
* Libraries: Pandas, NumPy, Scikit-learn, NLTK / spaCy
* ML Techniques: TF-IDF, Logistic Regression, Cosine Similarity
* Backend: Flask / FastAPI (integration-ready)
* Frontend: HTML, CSS, JavaScript (or any modern framework)

---

## Dataset Strategy

Due to the lack of publicly available labeled hiring datasets, CandiSight uses a hybrid data strategy:

* Real-world resume and job description text
* Synthetic label generation based on skill overlap thresholds
* Resume–JD pairing to expand dataset size

This approach mirrors real ATS logic and ensures explainability.

---

## Evaluation Metrics

* Classification: Accuracy, Precision, Recall, F1-score
* ATS Score: Percentage-based interpretability

F1-score is prioritized due to class imbalance in hiring scenarios.

---

## Future Enhancements

* Semantic embeddings (BERT/Sentence Transformers)
* Bias detection and fairness metrics
* Multi-role job matching
* Resume parsing into structured fields
* Dashboard analytics for recruiters

---

## Team Roles (Example)

* AI/ML Engineer: Data preprocessing, model training, evaluation, and model export
* Backend Developer: API development and ML integration
* Frontend Developer: UI/UX and result visualization
* System Integrator: End-to-end testing and deployment support

---

## Hackathon Readiness

* Scope-limited and achievable in 2 days of coding
* Fully explainable ML approach
* Real-world impact and scalability
* Clear demo flow for presentation

---

## Conclusion

CandiSight demonstrates how practical machine learning and NLP can be applied to solve real hiring challenges. The project emphasizes explainability, efficiency, and usability, making it suitable for hackathons as well as real-world extensions.
