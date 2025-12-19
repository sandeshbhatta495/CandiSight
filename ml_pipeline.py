import os
import json
from ml.preprocessing import TextCleaner
from ml.feature_extraction import TFIDFVectorizer, CosineSimilarity
from ml.model_training import LogisticRegressionModel
from ml.evaluation import ModelMetrics
from ml.ats_scoring import ATSScorer
import config


class MLPipeline:
    """
    End-to-end ML pipeline for candidate evaluation
    """
    
    def __init__(self):
        """Initialize all pipeline components"""
        self.text_cleaner = TextCleaner()
        self.tfidf_vectorizer = TFIDFVectorizer()
        self.lr_model = LogisticRegressionModel()
        self.ats_scorer = ATSScorer(config.SKILL_DICTIONARY_PATH)
        self.metrics = ModelMetrics()
    
    def load_models(self):
        """Load pre-trained models from disk"""
        if os.path.exists(config.TFIDF_MODEL_PATH):
            self.tfidf_vectorizer.load(config.TFIDF_MODEL_PATH)
            print("✓ Loaded TF-IDF vectorizer")
        
        if os.path.exists(config.LOGISTIC_REGRESSION_MODEL_PATH):
            self.lr_model.load(config.LOGISTIC_REGRESSION_MODEL_PATH)
            print("✓ Loaded Logistic Regression model")
    
    def save_models(self):
        """Save trained models to disk"""
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        self.tfidf_vectorizer.save(config.TFIDF_MODEL_PATH)
        self.lr_model.save(config.LOGISTIC_REGRESSION_MODEL_PATH)
        print("✓ Models saved successfully")
    
    def preprocess(self, resume_text, job_description):
        """
        Preprocess resume and job description
        
        Args:
            resume_text (str): Raw resume text
            job_description (str): Raw job description
            
        Returns:
            tuple: (cleaned_resume, cleaned_job_description)
        """
        cleaned_resume = self.text_cleaner.clean_text(resume_text)
        cleaned_job = self.text_cleaner.clean_text(job_description)
        return cleaned_resume, cleaned_job
    
    def evaluate_candidate(self, resume_text, job_description):
        """
        Complete evaluation pipeline for a single candidate
        
        Args:
            resume_text (str): Resume text
            job_description (str): Job description
            
        Returns:
            dict: Evaluation results with score, prediction, and ATS breakdown
        """
        # Preprocess
        cleaned_resume, cleaned_job = self.preprocess(resume_text, job_description)
        
        # Feature extraction
        resume_vector = self.tfidf_vectorizer.transform([cleaned_resume])
        job_vector = self.tfidf_vectorizer.transform([cleaned_job])
        
        # ML Prediction
        prediction = self.lr_model.predict(resume_vector)[0]
        probabilities = self.lr_model.predict_proba(resume_vector)[0]
        confidence = max(probabilities) * 100
        
        # ATS Scoring
        ats_result = self.ats_scorer.compute_ats_score(
            cleaned_resume,
            cleaned_job,
            ml_prediction_score=confidence,
            ml_weight=0.4
        )
        
        # Similarity
        similarity = CosineSimilarity.compute_similarity(resume_vector, job_vector)
        
        return {
            'prediction': 'GOOD_FIT' if prediction == 1 else 'NOT_A_FIT',
            'confidence': round(confidence, 2),
            'similarity_score': round(similarity * 100, 2),
            'ats_score': ats_result['overall_score'],
            'ats_rating': ats_result['rating'],
            'skill_match': ats_result['skill_match'],
            'keyword_overlap': ats_result['keyword_overlap'],
            'details': {
                'cleaned_resume_snippet': cleaned_resume[:200],
                'cleaned_job_snippet': cleaned_job[:200]
            }
        }
    
    def batch_evaluate(self, resumes, job_description):
        """
        Evaluate multiple candidates against one job description
        
        Args:
            resumes (list): List of resume texts
            job_description (str): Job description
            
        Returns:
            list: Evaluation results for each candidate
        """
        results = []
        for i, resume in enumerate(resumes):
            result = self.evaluate_candidate(resume, job_description)
            result['candidate_id'] = i + 1
            results.append(result)
        
        # Sort by ATS score
        results = sorted(results, key=lambda x: x['ats_score'], reverse=True)
        return results


# Example usage
if __name__ == "__main__":
    pipeline = MLPipeline()
    
    # Example texts
    sample_resume = """
    John Doe
    Senior Software Engineer
    
    Experience:
    - 5 years Python development
    - React and JavaScript
    - PostgreSQL and MongoDB
    - AWS and Docker
    
    Skills: Python, JavaScript, React, PostgreSQL, Docker, AWS
    """
    
    sample_job = """
    Senior Full Stack Developer
    Required Skills:
    - Python
    - React
    - PostgreSQL
    - Docker
    - AWS
    """
    
    # Evaluate
    result = pipeline.evaluate_candidate(sample_resume, sample_job)
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in result.items():
        print(f"{key}: {value}")
