"""
Text Cleaning and Preprocessing Module

Implements text preprocessing pipeline:
- Lowercase conversion
- Punctuation removal
- Stopword removal
- Lemmatization/Tokenization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextCleaner:
    """
    Text cleaning and preprocessing class
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Complete text cleaning pipeline
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Step 1: Lowercase
        text = text.lower()
        
        # Step 2: Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Step 3: Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Step 4: Remove special characters and digits (keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Step 5: Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 6: Tokenize
        tokens = word_tokenize(text)
        
        # Step 7: Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Step 8: Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Join back to string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens
        """
        cleaned = self.clean_text(text)
        return cleaned.split()
    
    def extract_keywords(self, text, skill_dictionary=None):
        """
        Extract relevant keywords from text
        
        Args:
            text (str): Text to extract keywords from
            skill_dictionary (list): List of important skills/keywords
            
        Returns:
            list: Extracted keywords
        """
        tokens = self.tokenize(text)
        
        if skill_dictionary:
            keywords = [token for token in tokens if token in skill_dictionary]
        else:
            keywords = tokens
        
        return keywords
