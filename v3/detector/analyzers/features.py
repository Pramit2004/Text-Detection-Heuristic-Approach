import re
import math
from collections import Counter
import numpy as np
from typing import List, Tuple  # Add for type hints

class FeatureExtractor:
    """
    Extracts various features for analysis
    """
    
    def __init__(self):
        self.stop_words = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from'
        ])
        
    def extract_all_features(self, text):
        """
        Extract comprehensive feature set
        """
        features = {}
        
        # Basic statistics
        features.update(self.extract_basic_stats(text))
        
        # Lexical features
        features.update(self.extract_lexical_features(text))
        
        # Syntactic features
        features.update(self.extract_syntactic_features(text))
        
        # Semantic features
        features.update(self.extract_semantic_features(text))
        
        # Stylistic features
        features.update(self.extract_stylistic_features(text))
        
        return features
    
    def extract_basic_stats(self, text):
        """
        Extract basic statistical features
        """
        # Word count
        words = text.split()
        word_count = len(words)
        
        # Character count
        char_count = len(text)
        
        # Sentence count
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # Average word length
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # Average sentence length (in words)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2)
        }
    
    def extract_lexical_features(self, text):
        """
        Extract lexical (vocabulary) features
        """
        words = text.lower().split()
        words = [w.strip('.,!?;:()[]{}"\'') for w in words]
        words = [w for w in words if w and w not in self.stop_words]
        
        if not words:
            return {
                'type_token_ratio': 0,
                'hapax_legomena': 0,
                'vocabulary_richness': 0
            }
        
        # Type-Token Ratio (unique words / total words)
        unique_words = set(words)
        type_token_ratio = len(unique_words) / len(words)
        
        # Hapax Legomena (words that appear only once)
        word_counts = Counter(words)
        hapax = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax / len(words)
        
        # Vocabulary richness (Brunet's Index)
        # W = N ^ (V ^ -a) where a is usually 0.17
        try:
            brunet_index = len(words) ** (len(unique_words) ** -0.17)
        except:
            brunet_index = 0
        
        return {
            'type_token_ratio': round(type_token_ratio, 3),
            'hapax_legomena_ratio': round(hapax_ratio, 3),
            'vocabulary_richness': round(brunet_index, 2)
        }
    
    def extract_syntactic_features(self, text):
        """
        Extract syntactic (grammar) features
        """
        # Part of speech patterns (simplified)
        words = text.split()
        
        # Function word ratio (the, and, of, etc.)
        function_words = set(['the', 'and', 'of', 'to', 'in', 'that', 'is', 'was'])
        function_word_count = sum(1 for w in words if w.lower() in function_words)
        function_word_ratio = function_word_count / len(words) if words else 0
        
        # Punctuation density
        punctuation = re.findall(r'[.,!?;:]', text)
        punctuation_density = len(punctuation) / len(words) if words else 0
        
        # Capitalization ratio (proper nouns)
        capitalized = sum(1 for w in words if w and w[0].isupper())
        capitalization_ratio = capitalized / len(words) if words else 0
        
        return {
            'function_word_ratio': round(function_word_ratio, 3),
            'punctuation_density': round(punctuation_density, 3),
            'capitalization_ratio': round(capitalization_ratio, 3)
        }
    
    def extract_semantic_features(self, text):
        """
        Extract semantic (meaning) features
        """
        # Pronoun usage
        pronouns = {
            'first_person': ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our'],
            'second_person': ['you', 'your', 'yours', 'yourself'],
            'third_person': ['he', 'him', 'his', 'she', 'her', 'it', 'they', 'them']
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        pronoun_counts = {}
        for category, pronoun_list in pronouns.items():
            count = sum(text_lower.count(pronoun) for pronoun in pronoun_list)
            pronoun_counts[category] = count
        
        # Sentiment words (simplified)
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic'])
        negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'poor', 'worst'])
        
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        
        return {
            'first_person_pronouns': pronoun_counts.get('first_person', 0),
            'second_person_pronouns': pronoun_counts.get('second_person', 0),
            'third_person_pronouns': pronoun_counts.get('third_person', 0),
            'positive_word_count': positive_count,
            'negative_word_count': negative_count
        }
    
    def extract_stylistic_features(self, text):
        """
        Extract stylistic (writing style) features
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return {
                'sentence_length_variation': 0,
                'complex_sentence_ratio': 0,
                'transition_word_density': 0
            }
        
        # Sentence length variation
        lengths = [len(s.split()) for s in sentences]
        length_variation = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
        
        # Complex sentences (containing commas or conjunctions)
        complex_count = sum(1 for s in sentences if ',' in s or ' and ' in s or ' but ' in s)
        complex_ratio = complex_count / len(sentences)
        
        # Transition words
        transition_words = set(['however', 'therefore', 'furthermore', 'moreover', 
                               'consequently', 'additionally', 'nevertheless'])
        transition_count = sum(text.lower().count(tw) for tw in transition_words)
        transition_density = transition_count / len(sentences)
        
        return {
            'sentence_length_variation': round(length_variation, 3),
            'complex_sentence_ratio': round(complex_ratio, 3),
            'transition_word_density': round(transition_density, 3)
        }
    
    def get_feature_vector(self, text):
        """
        Return all features as a flat dictionary (for ML models if needed)
        """
        features = self.extract_all_features(text)
        
        # Flatten any nested dictionaries
        flat_features = {}
        for key, value in features.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_features[f"{key}_{subkey}"] = subvalue
            else:
                flat_features[key] = value
        
        return flat_features