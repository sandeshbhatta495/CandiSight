"""
ATS Compatibility Scoring Module

Rule-based scoring logic to compute ATS compatibility score (0-100)
"""

import json


class ATSScorer:
    """
    Compute ATS compatibility score based on skill and keyword matching
    """
    
    def __init__(self, skill_dictionary_path=None):
        """
        Initialize ATS Scorer
        
        Args:
            skill_dictionary_path (str): Path to skill dictionary JSON file
        """
        self.skill_dictionary = self._load_skill_dictionary(skill_dictionary_path)
    
    def _load_skill_dictionary(self, filepath):
        """
        Load skill dictionary from JSON file
        
        Args:
            filepath (str): Path to skill dictionary
            
        Returns:
            dict: Skill dictionary
        """
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                print(f"Skill dictionary not found at {filepath}")
                return self._default_skills()
        return self._default_skills()
    
    def _default_skills(self):
        """
        Default skill dictionary
        
        Returns:
            dict: Default skills by category
        """
        return {
            'programming': ['python', 'java', 'javascript', 'c++', 'csharp', 'ruby', 'go', 'rust'],
            'web': ['react', 'angular', 'vue', 'html', 'css', 'nodejs', 'django', 'flask'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn'],
            'data': ['data analysis', 'pandas', 'numpy', 'analytics', 'excel']
        }
    
    def compute_skill_match(self, resume_text, job_description):
        """
        Compute skill match score
        
        Args:
            resume_text (str): Cleaned resume text
            job_description (str): Job description text
            
        Returns:
            float: Skill match percentage (0-100)
        """
        resume_text = resume_text.lower()
        job_description = job_description.lower()
        
        # Extract required skills from job description
        required_skills = self._extract_skills(job_description)
        
        # Extract available skills from resume
        available_skills = self._extract_skills(resume_text)
        
        if not required_skills:
            return 50.0  # Default if no skills found
        
        # Calculate match percentage
        matched = len(set(required_skills) & set(available_skills))
        match_percentage = (matched / len(required_skills)) * 100
        
        return min(match_percentage, 100.0)
    
    def compute_keyword_overlap(self, resume_text, job_description):
        """
        Compute keyword overlap score
        
        Args:
            resume_text (str): Resume text
            job_description (str): Job description text
            
        Returns:
            float: Keyword overlap percentage (0-100)
        """
        resume_words = set(resume_text.lower().split())
        job_words = set(job_description.lower().split())
        
        if not job_words:
            return 50.0
        
        overlap = len(resume_words & job_words)
        overlap_percentage = (overlap / len(job_words)) * 100
        
        return min(overlap_percentage, 100.0)
    
    def compute_ats_score(self, resume_text, job_description, 
                         ml_prediction_score=None, ml_weight=0.4):
        """
        Compute overall ATS compatibility score
        
        Args:
            resume_text (str): Resume text
            job_description (str): Job description
            ml_prediction_score (float): ML model's confidence (0-100)
            ml_weight (float): Weight for ML score
            
        Returns:
            dict: ATS score breakdown
        """
        # Rule-based scores
        skill_match = self.compute_skill_match(resume_text, job_description)
        keyword_overlap = self.compute_keyword_overlap(resume_text, job_description)
        
        # Default ATS if no ML score provided
        if ml_prediction_score is None:
            ats_score = (skill_match * 0.6 + keyword_overlap * 0.4)
        else:
            rule_score = (skill_match * 0.6 + keyword_overlap * 0.4)
            ats_score = (rule_score * (1 - ml_weight) + ml_prediction_score * ml_weight)
        
        return {
            'overall_score': round(ats_score, 2),
            'skill_match': round(skill_match, 2),
            'keyword_overlap': round(keyword_overlap, 2),
            'ml_score': ml_prediction_score,
            'rating': self._get_rating(ats_score)
        }
    
    def _extract_skills(self, text):
        """
        Extract skills from text
        
        Args:
            text (str): Text to extract skills from
            
        Returns:
            list: Extracted skills
        """
        text = text.lower()
        found_skills = []
        
        for category, skills in self.skill_dictionary.items():
            for skill in skills:
                if skill in text:
                    found_skills.append(skill)
        
        return found_skills
    
    def _get_rating(self, score):
        """
        Get qualitative rating for score
        
        Args:
            score (float): ATS score
            
        Returns:
            str: Rating (Excellent, Good, Fair, Poor)
        """
        if score >= 80:
            return 'EXCELLENT_FIT'
        elif score >= 60:
            return 'GOOD_FIT'
        elif score >= 40:
            return 'FAIR_FIT'
        else:
            return 'POOR_FIT'
