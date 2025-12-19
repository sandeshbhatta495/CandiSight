"""
Model Evaluation Metrics

Computes and reports various model performance metrics
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)
import numpy as np


class ModelMetrics:
    """
    Compute and report model evaluation metrics
    """
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba=None):
        """
        Compute comprehensive metrics
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
            y_proba (list): Predicted probabilities (optional)
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    @staticmethod
    def print_classification_report(y_true, y_pred):
        """
        Print detailed classification report
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
        """
        print(classification_report(y_true, y_pred, target_names=['No Fit', 'Good Fit']))
    
    @staticmethod
    def print_confusion_matrix(y_true, y_pred):
        """
        Print confusion matrix
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(f"True Negatives:  {cm[0][0]} | False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]} | True Positives:  {cm[1][1]}")
