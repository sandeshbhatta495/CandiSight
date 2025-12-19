# CandiSight - AI/ML Project Structure Summary

## ✅ Complete Project Structure Created

The AI/ML section of CandiSight has been fully structured and is ready for development. Below is the complete organization:

---

## 📁 Directory Structure

```
CandiSight/
│
├── 📂 ml/                                    # Main ML Module
│   ├── __init__.py                          # Module initialization
│   ├── ats_scoring.py                       # ATS compatibility scoring logic
│   │
│   ├── 📂 preprocessing/                    # Text Preprocessing
│   │   ├── __init__.py
│   │   └── text_cleaner.py                 # Text cleaning, tokenization, lemmatization
│   │
│   ├── 📂 feature_extraction/               # Feature Engineering
│   │   ├── __init__.py
│   │   ├── tfidf_vectorizer.py             # TF-IDF vectorization
│   │   └── similarity.py                    # Cosine similarity computation
│   │
│   ├── 📂 model_training/                   # ML Model Training
│   │   ├── __init__.py
│   │   └── logistic_regression_model.py    # Logistic regression classifier
│   │
│   └── 📂 evaluation/                       # Model Evaluation
│       ├── __init__.py
│       └── metrics.py                       # Evaluation metrics (accuracy, F1, etc.)
│
├── 📂 data/                                 # Data Management
│   ├── skill_dictionary.json                # Pre-defined skills by category
│   │
│   ├── 📂 raw/                              # Raw Data (to be populated)
│   │   ├── 📂 resumes/                      # Resume documents
│   │   │   └── .gitkeep
│   │   └── 📂 job_descriptions/             # Job description documents
│   │       └── .gitkeep
│   │
│   └── 📂 processed/                        # Processed Data
│       ├── labeled_data.csv                 # Training data (to be generated)
│       └── .gitkeep
│
├── 📂 models/                               # Trained Model Artifacts
│   ├── tfidf_vectorizer.pkl                 # Saved TF-IDF vectorizer
│   ├── logistic_regression_model.pkl        # Saved ML model
│   ├── model_metadata.json                  # Model information
│   └── .gitkeep
│
├── 📂 notebooks/                            # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb         # Feature creation and testing
│   ├── 03_model_training.ipynb              # Model training pipeline
│   └── .gitkeep
│
├── 📄 config.py                             # Centralized configuration
├── 📄 ml_pipeline.py                        # Main orchestrator (combines all components)
├── 📄 requirements.txt                      # Python dependencies
├── 📄 ML_SETUP_GUIDE.md                     # Detailed setup and usage guide
├── 📄 .gitignore                            # Git ignore rules
└── 📄 Readme.md                             # Main project README
```

---

## 🏗️ Module Organization

### 1. **ml/preprocessing/** - Text Preprocessing
   - **Purpose**: Clean and normalize text data
   - **Files**:
     - `text_cleaner.py`: TextCleaner class
   - **Features**:
     - Lowercase conversion
     - Punctuation & special character removal
     - Stopword removal
     - Lemmatization
     - Tokenization
     - Keyword extraction

### 2. **ml/feature_extraction/** - Feature Engineering
   - **Purpose**: Convert text to numerical features
   - **Files**:
     - `tfidf_vectorizer.py`: TF-IDF vectorization
     - `similarity.py`: Similarity metrics
   - **Features**:
     - TF-IDF vectorization
     - Cosine similarity computation
     - Batch similarity ranking

### 3. **ml/model_training/** - ML Model
   - **Purpose**: Train and use predictive model
   - **Files**:
     - `logistic_regression_model.py`: Classification model
   - **Features**:
     - Logistic regression classifier
     - Probability predictions
     - Model persistence (save/load)

### 4. **ml/evaluation/** - Model Evaluation
   - **Purpose**: Assess model performance
   - **Files**:
     - `metrics.py`: Evaluation metrics
   - **Features**:
     - Accuracy, precision, recall, F1-score
     - Confusion matrix
     - ROC-AUC
     - Classification reports

### 5. **ml/ats_scoring.py** - ATS Scoring
   - **Purpose**: Generate ATS compatibility scores
   - **Features**:
     - Skill matching
     - Keyword overlap calculation
     - Rule-based scoring
     - ML-based scoring integration
     - Rating generation (EXCELLENT, GOOD, FAIR, POOR)

---

## 📊 Data Organization

### Raw Data (`data/raw/`)
- **resumes/**: Directory for resume text files
- **job_descriptions/**: Directory for job description files

### Processed Data (`data/processed/`)
- **labeled_data.csv**: Training dataset with resume, job description, and label

### Skill Dictionary (`data/skill_dictionary.json`)
- Categorized skills dictionary
- Used for skill extraction and matching
- Categories: programming, web, databases, cloud, ML, data, other

---

## 🔧 Core Components

### TextCleaner
```python
cleaner = TextCleaner()
cleaned_text = cleaner.clean_text(raw_text)
tokens = cleaner.tokenize(text)
keywords = cleaner.extract_keywords(text, skill_dict)
```

### TFIDFVectorizer
```python
vectorizer = TFIDFVectorizer(max_features=1000)
vectorizer.fit(training_texts)
vectors = vectorizer.transform(new_texts)
```

### CosineSimilarity
```python
similarity = CosineSimilarity.compute_similarity(vector1, vector2)
ranked = CosineSimilarity.rank_by_similarity(query, candidates)
```

### LogisticRegressionModel
```python
model = LogisticRegressionModel()
model.train(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### ModelMetrics
```python
metrics = ModelMetrics.compute_metrics(y_true, y_pred)
ModelMetrics.print_classification_report(y_true, y_pred)
```

### ATSScorer
```python
scorer = ATSScorer('data/skill_dictionary.json')
ats_result = scorer.compute_ats_score(resume, job_description)
```

---

## 🚀 Main Pipeline (ml_pipeline.py)

The `MLPipeline` class orchestrates all components:

```python
pipeline = MLPipeline()

# Single evaluation
result = pipeline.evaluate_candidate(resume, job_description)

# Batch evaluation
results = pipeline.batch_evaluate(resumes, job_description)

# Save/load models
pipeline.save_models()
pipeline.load_models()
```

---

## 📋 Configuration (config.py)

Central configuration file with:
- Directory paths
- ML hyperparameters
- ATS scoring weights
- Score thresholds
- Dataset settings
- Logging configuration

---

## 📦 Dependencies (requirements.txt)

Key packages:
- **numpy, pandas**: Data manipulation
- **scikit-learn**: ML algorithms
- **nltk, spacy**: NLP processing
- **PyPDF2, python-docx**: File parsing
- **flask**: Backend API
- **jupyter**: Notebooks

---

## 🎯 Development Roadmap

### Phase 1: Data Preparation ✅
- [x] Create folder structure
- [x] Skill dictionary
- [ ] Collect resume samples
- [ ] Collect job descriptions
- [ ] Generate labeled training data

### Phase 2: ML Development ✅
- [x] Text preprocessing module
- [x] Feature extraction module
- [x] Model training module
- [x] Evaluation module
- [x] ATS scoring module
- [ ] Train model on data
- [ ] Evaluate performance
- [ ] Save trained models

### Phase 3: Integration
- [ ] Create Flask API endpoints
- [ ] Connect ML pipeline to backend
- [ ] Create frontend interface
- [ ] Deploy application

---

## 📚 Usage Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
- Add resume files to `data/raw/resumes/`
- Add job descriptions to `data/raw/job_descriptions/`
- Create `data/processed/labeled_data.csv`

### 3. Run Evaluation
```python
from ml_pipeline import MLPipeline

pipeline = MLPipeline()
result = pipeline.evaluate_candidate(resume_text, job_description)
print(result)
```

---

## 📖 Documentation

- **ML_SETUP_GUIDE.md**: Detailed setup and usage instructions
- **Readme.md**: Main project documentation
- **Code docstrings**: Inline documentation in each module

---

## ✨ Key Features

✅ **Modular Architecture**: Each component is independent and reusable
✅ **Text Preprocessing**: Comprehensive text cleaning pipeline
✅ **Feature Extraction**: TF-IDF with cosine similarity
✅ **ML Classification**: Logistic regression for fit prediction
✅ **ATS Scoring**: Rule-based + ML hybrid scoring
✅ **Model Persistence**: Save/load trained models
✅ **Batch Processing**: Evaluate multiple candidates
✅ **Comprehensive Metrics**: F1, accuracy, ROC-AUC, etc.
✅ **Well Documented**: Docstrings and guides throughout
✅ **Configuration Driven**: Easy to adjust parameters

---

## 🎯 Next Steps

1. **Add Training Data**
   - Collect real resumes and job descriptions
   - Create labeled dataset in CSV format

2. **Train Models**
   - Run Jupyter notebooks for training
   - Evaluate model performance

3. **Deploy**
   - Create Flask API
   - Build frontend interface
   - Deploy to production

4. **Iterate**
   - Monitor performance
   - Collect feedback
   - Improve models

---

## 📞 Support

For questions or issues, refer to:
- **ML_SETUP_GUIDE.md** for detailed instructions
- **Code docstrings** for API documentation
- **Readme.md** for project overview

---

**Ready to start development! 🚀**

The structure is set up and ready for your team to:
1. Add training data
2. Train models
3. Evaluate performance
4. Integrate with backend API
5. Deploy to production
