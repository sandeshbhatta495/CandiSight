"""
Logistic Regression Model for Job Fit Prediction

Binary classification model to predict whether a candidate is a good fit for a job
"""

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


class LogisticRegressionModel:
    """
    Logistic Regression classifier for candidate-job fit prediction
    """
    
    def __init__(self, random_state=42, max_iter=1000):
        """
        Initialize model
        
        Args:
            random_state (int): Random state for reproducibility
            max_iter (int): Maximum iterations for solver
        """
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver='lbfgs'
        )
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features (TF-IDF vectors)
            y_train: Training labels (0 or 1)
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: Feature vectors
            
        Returns:
            np.array: Predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Feature vectors
            
        Returns:
            np.array: Predicted probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def get_confidence(self, X):
        """
        Get confidence scores for predictions
        
        Args:
            X: Feature vectors
            
        Returns:
            list: Confidence scores (0 to 1)
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1).tolist()
    
    def save(self, filepath):
        """
        Save model to disk
        
        Args:
            filepath (str): Path to save model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):
        """
        Load model from disk
        
        Args:
            filepath (str): Path to load model from
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
