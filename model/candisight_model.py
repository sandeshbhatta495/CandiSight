import re
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import joblib
import os
from typing import Tuple, Dict, List

class TextPreprocessor:
    """Handles text preprocessing for resumes and job descriptions."""
    
    def __init__(self):
        # Load English language model for lemmatization
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, special characters, and numbers
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Lemmatization and remove stopwords
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)

class ATSModel:
    """Main ATS model for resume-job matching."""
    
    def __init__(self, model_path: str = None):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.preprocessor = TextPreprocessor()
        self.skills_vocab = set()
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'saved_models')
        
        # Create directory for saved models if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
    
    def preprocess_data(self, texts: List[str]) -> List[str]:
        """Preprocess a list of text documents."""
        return [self.preprocessor.clean_text(text) for text in texts]
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract TF-IDF features from text."""
        processed_texts = self.preprocess_data(texts)
        return self.vectorizer.transform(processed_texts)
    
    def calculate_similarity(self, resume_text: str, job_desc: str) -> float:
        """Calculate cosine similarity between resume and job description."""
        processed_texts = self.preprocess_data([resume_text, job_desc])
        tfidf_matrix = self.vectorizer.transform(processed_texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]
    
    def calculate_ats_score(self, resume_text: str, job_desc: str) -> Dict[str, float]:
        """Calculate ATS score based on skill matching and similarity."""
        # Basic implementation - can be enhanced with more sophisticated scoring
        similarity = self.calculate_similarity(resume_text, job_desc)
        
        # Simple keyword matching (can be expanded)
        resume_tokens = set(self.preprocessor.clean_text(resume_text).split())
        job_tokens = set(self.preprocessor.clean_text(job_desc).split())
        
        # Calculate keyword overlap
        if job_tokens:
            keyword_match = len(resume_tokens.intersection(job_tokens)) / len(job_tokens)
        else:
            keyword_match = 0.0
        
        # Combine scores (weights can be adjusted)
        ats_score = (0.6 * similarity) + (0.4 * keyword_match)
        ats_score = max(0.0, min(1.0, ats_score)) * 100  # Convert to percentage
        
        return {
            'ats_score': round(ats_score, 2),
            'similarity_score': round(similarity * 100, 2),
            'keyword_match': round(keyword_match * 100, 2)
        }
    
    def predict_fit(self, resume_text: str, job_desc: str) -> Dict:
        """Predict if a candidate is a good fit for the job."""
        # Calculate features
        features = self.extract_features([resume_text + " " + job_desc])
        
        # Get prediction probability
        proba = self.classifier.predict_proba(features)[0]
        prediction = int(proba[1] > 0.5)  # Threshold can be adjusted
        
        # Get ATS score
        ats_scores = self.calculate_ats_score(resume_text, job_desc)
        
        return {
            'prediction': prediction,
            'probability': float(proba[1]),
            'ats_scores': ats_scores,
            'explanation': self._generate_explanation(prediction, ats_scores)
        }
    
    def _generate_explanation(self, prediction: int, ats_scores: Dict) -> str:
        """Generate human-readable explanation of the prediction."""
        if prediction == 1:
            return (f"This candidate is a good fit with an ATS score of {ats_scores['ats_score']}%. "
                   f"The resume shows strong alignment with the job requirements.")
        else:
            return (f"This candidate may not be the best fit (ATS score: {ats_scores['ats_score']}%). "
                   f"Consider candidates with better skill alignment.")
    
    def train(self, X_train: List[str], y_train: List[int]) -> None:
        """Train the model on labeled data."""
        # Preprocess and vectorize training data
        X_processed = self.preprocess_data(X_train)
        X_tfidf = self.vectorizer.fit_transform(X_processed)
        
        # Train classifier
        self.classifier.fit(X_tfidf, y_train)
    
    def save_model(self, model_name: str = 'candisight_model') -> None:
        """Save the trained model and vectorizer."""
        model_file = os.path.join(self.model_path, f'{model_name}.pkl')
        vectorizer_file = os.path.join(self.model_path, f'{model_name}_vectorizer.pkl')
        
        joblib.dump(self.classifier, model_file)
        joblib.dump(self.vectorizer, vectorizer_file)
    
    def load_model(self, model_name: str = 'candisight_model') -> None:
        """Load a trained model and vectorizer."""
        model_file = os.path.join(self.model_path, f'{model_name}.pkl')
        vectorizer_file = os.path.join(self.model_path, f'{model_name}_vectorizer.pkl')
        
        if os.path.exists(model_file) and os.path.exists(vectorizer_file):
            self.classifier = joblib.load(model_file)
            self.vectorizer = joblib.load(vectorizer_file)
            return True
        return False

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = ATSModel()
    
    # Example usage with dummy data
    resume = """
    John Doe
    Senior Data Scientist
    
    Skills: Python, Machine Learning, Deep Learning, NLP, TensorFlow, PyTorch
    Experience: 5+ years in AI and data science
    Education: MS in Computer Science
    """
    
    job_description = """
    We are looking for a Senior Data Scientist with:
    - Strong experience in Python and Machine Learning
    - Knowledge of Deep Learning frameworks (TensorFlow/PyTorch)
    - Experience with Natural Language Processing (NLP)
    - 3+ years of relevant experience
    """
    
    # Make prediction
    result = model.predict_fit(resume, job_description)
    print("Prediction:", "Good Fit" if result['prediction'] == 1 else "Not a Good Fit")
    print("ATS Score:", result['ats_scores']['ats_score'])
    print("Explanation:", result['explanation'])
