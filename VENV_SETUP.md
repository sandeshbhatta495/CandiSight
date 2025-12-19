# Virtual Environment Setup Guide

## Quick Setup (Windows)

### Option 1: Using Batch Script (Easiest)
```bash
setup.bat
```
This will automatically:
1. Create virtual environment
2. Activate it
3. Install all dependencies

### Option 2: Using Python Script
```bash
python setup.py
```

### Option 3: Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat

# Install requirements
pip install -r requirements.txt
```

---

## Quick Setup (Mac/Linux)

### Option 1: Using Python Script
```bash
python setup.py
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## After Activation

Once activated, your terminal/command prompt will show `(venv)` prefix:

**Windows:**
```
(venv) C:\Users\DELL\OneDrive\Desktop\Hackathon>
```

**Mac/Linux:**
```
(venv) ~/Hackathon$
```

---

## Verify Installation

```bash
python --version
pip list
```

You should see all packages from requirements.txt installed.

---

## Deactivate Virtual Environment

```bash
deactivate
```

---

## Project Structure

```
Hackathon/
├── venv/                          # Virtual environment (created by setup)
│   ├── Scripts/                   # Windows executables
│   ├── Lib/                       # Python packages
│   └── pyvenv.cfg                # Configuration
├── ml/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── ats_scorer.py
│   ├── pipeline.py
│   └── utils.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── models/
│   │   ├── model.pkl
│   │   └── vectorizer.pkl
│   └── sample/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_ats_scorer.py
├── requirements.txt
├── setup.bat                      # Windows setup script
├── setup.py                       # Python setup script
├── Readme.md                      # Project documentation
└── PROJECT_STRUCTURE.md           # This file
```

---

## Installed Packages

Core packages that will be installed:
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - ML algorithms
- **nltk** - NLP preprocessing
- **spacy** - Advanced NLP (optional)
- **jupyter** - For notebooks
- **flask** - Backend API (for deployment)
- **python-dotenv** - Environment variables

See `requirements.txt` for complete list with versions.

---

## Troubleshooting

### Issue: "python command not found"
**Solution:** Make sure Python is installed and added to PATH. Check with:
```bash
python --version
```

### Issue: "Permission denied" when running setup.bat
**Solution:** Run Command Prompt as Administrator, then run setup.bat

### Issue: Packages not installing
**Solution:** Try upgrading pip first:
```bash
python -m pip install --upgrade pip
```

### Issue: Virtual environment not activating
**Solution:** Check the path and try:
```bash
cd /d C:\Users\DELL\OneDrive\Desktop\Hackathon
venv\Scripts\activate.bat
```

---

## Next Steps

1. ✅ Create virtual environment (you're here!)
2. 📊 Prepare datasets (place in `data/raw/`)
3. 🤖 Train ML model (`ml/model_training.py`)
4. ⚙️ Test pipeline (`ml/pipeline.py`)
5. 🎯 Export model to `data/models/`
6. 🚀 Deploy with Flask backend

---

## Tips

- Always work **within the activated virtual environment**
- Keep venv out of version control (it's in .gitignore)
- Install new packages with `pip install package_name`
- Save new dependencies with `pip freeze > requirements.txt`
- For Jupyter: `jupyter notebook` to start notebook server

---

**Setup complete! Happy coding! 🚀**
