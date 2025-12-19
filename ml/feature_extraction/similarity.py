"""
Similarity Computation Module

Computes similarity metrics between resume and job description vectors
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarity:
    """
    Cosine Similarity computation for comparing vectors
    """
    
    @staticmethod
    def compute_similarity(vector1, vector2):
        """
        Compute cosine similarity between two vectors
        
        Args:
            vector1: First vector (TF-IDF vector)
            vector2: Second vector (TF-IDF vector)
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Handle sparse matrices
        similarity = cosine_similarity(vector1, vector2)
        return float(similarity[0][0])
    
    @staticmethod
    def compute_batch_similarity(vectors1, vectors2):
        """
        Compute similarity between batches of vectors
        
        Args:
            vectors1: Batch of vectors 1
            vectors2: Batch of vectors 2
            
        Returns:
            np.array: Similarity matrix
        """
        return cosine_similarity(vectors1, vectors2)
    
    @staticmethod
    def rank_by_similarity(query_vector, candidate_vectors):
        """
        Rank candidates by similarity to query
        
        Args:
            query_vector: Query vector
            candidate_vectors: Candidate vectors
            
        Returns:
            list: Ranked indices with scores
        """
        similarities = cosine_similarity(query_vector, candidate_vectors)[0]
        ranked = sorted(
            enumerate(similarities),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked
