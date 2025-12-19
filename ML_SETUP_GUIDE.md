# CandiSight ML Module - Setup and Usage Guide

## Project Structure

```
CandiSight/
├── ml/                              # Main ML module
│   ├── __init__.py                 # Module initialization
│   ├── preprocessing/              # Text preprocessing
│   │   ├── __init__.py
│   │   └── text_cleaner.py        # Text cleaning and tokenization
│   ├── feature_extraction/         # Feature extraction
│   │   ├── __init__.py
│   │   ├── tfidf_vectorizer.py    # TF-IDF vectorization
│   │   └── similarity.py           # Similarity computation
│   ├── model_training/             # Model training
│   │   ├── __init__.py
│   │   └── logistic_regression_model.py
│   ├── evaluation/                 # Model evaluation
│   │   ├── __init__.py
│   │   └── metrics.py              # Evaluation metrics
│   └── ats_scoring.py              # ATS compatibility scoring
├── data/                            # Data directory
│   ├── raw/                        # Raw data
│   │   ├── resumes/                # Resume documents
│   │   └── job_descriptions/       # Job description documents
│   ├── processed/                  # Processed data
│   │   └── labeled_data.csv        # Labeled training data
│   └── skill_dictionary.json       # Skill dictionary
├── models/                          # Trained models
│   ├── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer
│   ├── logistic_regression_model.pkl
│   └── model_metadata.json
├── notebooks/                       # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── config.py                        # Configuration file
├── ml_pipeline.py                   # Main pipeline orchestrator
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data

Run this once to download required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 3. Prepare Data

1. Add resume texts to `data/raw/resumes/`
2. Add job descriptions to `data/raw/job_descriptions/`
3. Create `data/processed/labeled_data.csv` with columns: `resume`, `job_description`, `label`

### 4. Configure Parameters

Edit `config.py` to adjust:
- ML hyperparameters
- File paths
- Score thresholds
- Weights for ATS scoring

## Usage

### Quick Start: Single Candidate Evaluation

```python
from ml_pipeline import MLPipeline

# Initialize pipeline
pipeline = MLPipeline()
pipeline.load_models()

# Evaluate a candidate
resume = "Your resume text here..."
job_description = "Your job description here..."

result = pipeline.evaluate_candidate(resume, job_description)

print(f"ATS Score: {result['ats_score']}")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")
print(f"Rating: {result['ats_rating']}")
```

### Batch Evaluation: Multiple Candidates

```python
resumes = [resume1, resume2, resume3, ...]
job_description = "Your job description..."

results = pipeline.batch_evaluate(resumes, job_description)

# Results are sorted by ATS score (highest first)
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['ats_score']} - {result['ats_rating']}")
```

### Text Preprocessing

```python
from ml.preprocessing import TextCleaner

cleaner = TextCleaner()

# Clean text
cleaned = cleaner.clean_text("Your text here...")

# Tokenize
tokens = cleaner.tokenize("Your text here...")

# Extract keywords
keywords = cleaner.extract_keywords(
    "Your text here...",
    skill_dictionary=['python', 'react', 'javascript']
)
```

### Feature Extraction

```python
from ml.feature_extraction import TFIDFVectorizer, CosineSimilarity

# Initialize vectorizer
vectorizer = TFIDFVectorizer(max_features=1000)

# Fit on training data
vectorizer.fit(training_texts)

# Transform new texts
vectors = vectorizer.transform(new_texts)

# Compute similarity
similarity = CosineSimilarity.compute_similarity(vector1, vector2)
```

### Model Training

```python
from ml.model_training import LogisticRegressionModel
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_labels, test_size=0.2
)

# Train model
model = LogisticRegressionModel()
model.train(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Save model
model.save('models/logistic_regression_model.pkl')
```

### Model Evaluation

```python
from ml.evaluation import ModelMetrics

metrics = ModelMetrics.compute_metrics(y_true, y_pred, y_proba)

print(f"Accuracy: {metrics['accuracy']}")
print(f"F1-Score: {metrics['f1_score']}")
print(f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")

# Print reports
ModelMetrics.print_classification_report(y_true, y_pred)
ModelMetrics.print_confusion_matrix(y_true, y_pred)
```

### ATS Scoring

```python
from ml.ats_scoring import ATSScorer

scorer = ATSScorer('data/skill_dictionary.json')

# Compute ATS score
ats_result = scorer.compute_ats_score(
    resume_text,
    job_description,
    ml_prediction_score=85,
    ml_weight=0.4
)

print(f"Overall Score: {ats_result['overall_score']}")
print(f"Skill Match: {ats_result['skill_match']}%")
print(f"Rating: {ats_result['rating']}")
```

## Configuration

Edit `config.py` to customize:

```python
# TF-IDF settings
ML_CONFIG['tfidf'] = {
    'max_features': 1000,
    'ngram_range': (1, 2),
}

# Model settings
ML_CONFIG['logistic_regression'] = {
    'max_iter': 1000,
}

# ATS scoring weights
ATS_CONFIG = {
    'skill_match_weight': 0.6,
    'keyword_overlap_weight': 0.4,
    'ml_model_weight': 0.4,
    'rule_based_weight': 0.6
}

# Score thresholds
SCORE_THRESHOLDS = {
    'excellent': 80,
    'good': 60,
    'fair': 40,
    'poor': 0
}
```

## Data Preparation

### Creating Labeled Dataset

To create synthetic labeled data based on skill overlap:

```python
import pandas as pd
from ml.ats_scoring import ATSScorer

scorer = ATSScorer('data/skill_dictionary.json')

# Load resumes and job descriptions
resumes = [...]  # List of resume texts
jobs = [...]     # List of job description texts

# Generate labels based on skill overlap
data = []
for resume in resumes:
    for job in jobs:
        skill_match = scorer.compute_skill_match(resume, job)
        label = 1 if skill_match >= 60 else 0  # Good fit if > 60%
        data.append({
            'resume': resume,
            'job_description': job,
            'label': label
        })

df = pd.DataFrame(data)
df.to_csv('data/processed/labeled_data.csv', index=False)
```

## Evaluation Metrics

The model uses the following metrics:

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of all predicted positive, how many are truly positive
- **Recall**: Of all actual positive, how many were predicted
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

F1-Score is prioritized due to potential class imbalance in hiring scenarios.

## ATS Scoring

The ATS score (0-100) is computed as:

```
ATS_Score = (Rule_Based_Score * 0.6) + (ML_Model_Score * 0.4)

Where:
Rule_Based_Score = (Skill_Match * 0.6) + (Keyword_Overlap * 0.4)
```

### Rating Categories

- **EXCELLENT_FIT**: Score >= 80
- **GOOD_FIT**: Score >= 60 and < 80
- **FAIR_FIT**: Score >= 40 and < 60
- **POOR_FIT**: Score < 40

## Troubleshooting

### Issue: NLTK data not found

**Solution**: Run these commands

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Issue: Models not loading

**Solution**: Train and save models first

```python
# Train on your data
pipeline.train(training_data, training_labels)
pipeline.save_models()
```

### Issue: Low accuracy

**Solution**:
1. Check data quality and labeling
2. Increase training data size
3. Adjust hyperparameters in config.py
4. Try different preprocessing techniques

## Next Steps

1. Prepare your resume and job description datasets
2. Create labeled training data
3. Run model training notebooks
4. Evaluate model performance
5. Deploy to backend API
6. Integrate with frontend

## Support

For issues or questions, refer to the notebooks or README.md in the project root.
