"""
Configuration File for CandiSight ML Pipeline

Centralized configuration for all ML components
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

# Data files
RESUMES_DIR = os.path.join(RAW_DATA_DIR, 'resumes')
JOB_DESCRIPTIONS_DIR = os.path.join(RAW_DATA_DIR, 'job_descriptions')
SKILL_DICTIONARY_PATH = os.path.join(DATA_DIR, 'skill_dictionary.json')
LABELED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'labeled_data.csv')

# Model files
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
LOGISTIC_REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')

# ML Pipeline Configuration
ML_CONFIG = {
    'tfidf': {
        'max_features': 1000,
        'ngram_range': (1, 2),
        'min_df': 1,
        'max_df': 0.95
    },
    'logistic_regression': {
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'lbfgs'
    },
    'train_test_split': {
        'test_size': 0.2,
        'random_state': 42
    }
}

# ATS Scoring Configuration
ATS_CONFIG = {
    'skill_match_weight': 0.6,
    'keyword_overlap_weight': 0.4,
    'ml_model_weight': 0.4,
    'rule_based_weight': 0.6
}

# Scoring Thresholds
SCORE_THRESHOLDS = {
    'excellent': 80,
    'good': 60,
    'fair': 40,
    'poor': 0
}

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'candisight.log')

# Dataset Configuration
DATASET_CONFIG = {
    'synthetic_label_generation': True,
    'skill_overlap_threshold': 0.5,
    'min_skill_match_good_fit': 0.6,
    'max_skill_match_poor_fit': 0.3
}
