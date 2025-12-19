"""
Feature Extraction Module
Handles TF-IDF vectorization and similarity computation
"""

from .tfidf_vectorizer import TFIDFVectorizer
from .similarity import CosineSimilarity

__all__ = ['TFIDFVectorizer', 'CosineSimilarity']
