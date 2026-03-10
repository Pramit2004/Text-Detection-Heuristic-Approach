import math
import numpy as np
from collections import Counter
import json
import os

class PerplexityAnalyzer:
    """
    Uses pre-trained language model probabilities for perplexity calculation
    """
    
    def __init__(self, model_path=None):
        self.word_probabilities = self._load_base_probabilities()
        
    def _load_base_probabilities(self):
        """
        Load pre-computed word probabilities from a small dataset
        In production, you'd load from a file
        """
        # These are placeholder probabilities from common English
        return {
            'the': 0.061, 'be': 0.042, 'to': 0.035, 'of': 0.034,
            'and': 0.031, 'a': 0.030, 'in': 0.028, 'that': 0.025,
            'have': 0.021, 'i': 0.020, 'it': 0.019, 'for': 0.018,
            'not': 0.017, 'on': 0.016, 'with': 0.015, 'he': 0.014,
            'as': 0.013, 'you': 0.012, 'do': 0.011, 'at': 0.010
        }
    
    def calculate_token_perplexity(self, text):
        """
        Calculate perplexity using token probabilities
        """
        words = text.lower().split()
        if len(words) < 5:
            return 50.0
        
        log_prob_sum = 0
        unknown_words = 0
        
        for word in words:
            # Get probability, use small default for unknown words
            prob = self.word_probabilities.get(word, 0.0001)
            if prob <= 0.0001:
                unknown_words += 1
            
            log_prob = math.log2(prob)
            log_prob_sum += log_prob
        
        avg_log_prob = log_prob_sum / len(words)
        perplexity = 2 ** (-avg_log_prob)
        
        # Adjust for unknown words (creative writing = more unknown words)
        unknown_ratio = unknown_words / len(words)
        
        # Higher unknown word ratio suggests human writing
        human_adjustment = 1 + (unknown_ratio * 2)
        adjusted_perplexity = perplexity * human_adjustment
        
        # Normalize to 0-100
        # Typical range: 30-200
        normalized = min(100, max(0, (adjusted_perplexity - 30) / 1.7))
        
        return {
            'raw_perplexity': round(perplexity, 2),
            'adjusted_perplexity': round(adjusted_perplexity, 2),
            'unknown_word_ratio': round(unknown_ratio * 100, 2),
            'score': round(normalized, 2)
        }
    
    def calculate_ngram_perplexity(self, text, n=3):
        """
        Calculate n-gram based perplexity (better for detecting AI patterns)
        """
        words = text.lower().split()
        if len(words) < n + 1:
            return 50.0
        
        # Build n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        # Count n-gram frequencies
        ngram_counts = Counter(ngrams)
        
        # Calculate perplexity on n-gram level
        total_ngrams = len(ngrams)
        unique_ngrams = len(ngram_counts)
        
        # High unique n-gram ratio = creative = human
        unique_ratio = unique_ngrams / total_ngrams
        
        # AI tends to repeat n-grams more
        if unique_ratio > 0.7:
            return 85.0  # Very creative = human
        elif unique_ratio > 0.5:
            return 65.0  # Moderately creative
        elif unique_ratio > 0.3:
            return 40.0  # Some repetition = AI-like
        else:
            return 20.0  # Highly repetitive = AI-generated