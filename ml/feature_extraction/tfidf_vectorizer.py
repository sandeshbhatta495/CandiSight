"""
TF-IDF Vectorization Module

Converts text documents into numerical feature vectors using TF-IDF
"""

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TFIDFVectorizer:
    """
    TF-IDF Vectorizer wrapper for consistent feature extraction
    """
    
    def __init__(self, max_features=1000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF vectorizer
        
        Args:
            max_features (int): Maximum number of features
            ngram_range (tuple): N-gram range (min, max)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=1,
            max_df=0.95,
            stop_words='english'
        )
        self.is_fitted = False
    
    def fit(self, texts):
        """
        Fit the vectorizer on training data
        
        Args:
            texts (list): List of text documents
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts):
        """
        Transform texts to TF-IDF vectors
        
        Args:
            texts (list): List of text documents
            
        Returns:
            sparse matrix: TF-IDF feature vectors
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """
        Fit and transform in one step
        
        Args:
            texts (list): List of text documents
            
        Returns:
            sparse matrix: TF-IDF feature vectors
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self):
        """
        Get feature names from vectorizer
        
        Returns:
            list: Feature names
        """
        return self.vectorizer.get_feature_names_out().tolist()
    
    def save(self, filepath):
        """
        Save vectorizer to disk
        
        Args:
            filepath (str): Path to save vectorizer
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load(self, filepath):
        """
        Load vectorizer from disk
        
        Args:
            filepath (str): Path to load vectorizer from
        """
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
