"""
NLP Utilities - Complete implementation
Provides advanced NLP functions for text analysis
"""

import re
import math
import string
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Try to import NLTK with fallback
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Using basic NLP functionality.")

# Download required NLTK data if available
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

class NLPUtils:
    """
    Advanced NLP utilities for text analysis
    Provides functions for readability, POS tagging, sentiment, etc.
    """
    
    # Part of speech mappings (simplified)
    POS_MAPPING = {
        'NN': 'noun', 'NNS': 'noun', 'NNP': 'noun', 'NNPS': 'noun',
        'VB': 'verb', 'VBD': 'verb', 'VBG': 'verb', 'VBN': 'verb',
        'VBP': 'verb', 'VBZ': 'verb',
        'JJ': 'adjective', 'JJR': 'adjective', 'JJS': 'adjective',
        'RB': 'adverb', 'RBR': 'adverb', 'RBS': 'adverb',
        'PRP': 'pronoun', 'PRP$': 'pronoun',
        'DT': 'determiner', 'IN': 'preposition', 'CC': 'conjunction',
        'UH': 'interjection', 'CD': 'number', 'EX': 'existential'
    }
    
    # Readability score names and descriptions
    READABILITY_SCORES = {
        'flesch_reading_ease': {
            'name': 'Flesch Reading Ease',
            'description': 'Higher scores (60-70) indicate easier readability',
            'ranges': [
                (90, 100, 'Very Easy'),
                (80, 89, 'Easy'),
                (70, 79, 'Fairly Easy'),
                (60, 69, 'Standard'),
                (50, 59, 'Fairly Difficult'),
                (30, 49, 'Difficult'),
                (0, 29, 'Very Confusing')
            ]
        },
        'flesch_kincaid_grade': {
            'name': 'Flesch-Kincaid Grade Level',
            'description': 'US school grade level required to understand text',
            'ranges': [
                (0, 5, 'Elementary School'),
                (6, 8, 'Middle School'),
                (9, 12, 'High School'),
                (13, 16, 'College'),
                (17, 100, 'Graduate Level')
            ]
        },
        'gunning_fog': {
            'name': 'Gunning Fog Index',
            'description': 'Years of education needed to understand text',
            'ranges': [
                (0, 6, 'Easy'),
                (7, 12, 'Ideal'),
                (13, 17, 'Difficult'),
                (18, 100, 'Very Difficult')
            ]
        },
        'smog_index': {
            'name': 'SMOG Index',
            'description': 'Years of education needed (more accurate for longer texts)',
            'ranges': [
                (0, 6, 'Elementary'),
                (7, 12, 'Secondary'),
                (13, 16, 'College'),
                (17, 100, 'Graduate')
            ]
        },
        'coleman_liau': {
            'name': 'Coleman-Liau Index',
            'description': 'Based on characters per word and sentences',
            'ranges': [
                (0, 5, 'Elementary'),
                (6, 9, 'Middle School'),
                (10, 12, 'High School'),
                (13, 16, 'College'),
                (17, 100, 'Graduate')
            ]
        },
        'automated_readability': {
            'name': 'Automated Readability Index',
            'description': 'Characters per word and words per sentence',
            'ranges': [
                (0, 5, 'Elementary'),
                (6, 9, 'Middle School'),
                (10, 12, 'High School'),
                (13, 16, 'College'),
                (17, 100, 'Graduate')
            ]
        }
    }
    
    @staticmethod
    def get_readability_scores(text: str) -> Dict[str, Any]:
        """
        Calculate multiple readability scores
        Args:
            text: Input text
        Returns:
            Dictionary with various readability scores
        """
        # Get basic text statistics
        sentences = NLPUtils.get_sentences(text)
        words = NLPUtils.get_words(text)
        syllables = NLPUtils.count_syllables(text)
        characters = len(text)
        
        if not sentences or not words:
            return {
                'error': 'Text too short for readability analysis',
                'scores': {}
            }
        
        # Calculate base metrics
        sentence_count = len(sentences)
        word_count = len(words)
        syllable_count = syllables
        
        # Average per 100 words (for some formulas)
        sentences_per_100 = (sentence_count / word_count) * 100 if word_count > 0 else 0
        syllables_per_100 = (syllable_count / word_count) * 100 if word_count > 0 else 0
        characters_per_100 = (characters / word_count) * 100 if word_count > 0 else 0
        
        # Count complex words (3+ syllables)
        complex_words = NLPUtils.count_complex_words(text)
        
        # Calculate scores
        scores = {}
        
        # 1. Flesch Reading Ease
        # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        if word_count > 0 and sentence_count > 0:
            flesch = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
            scores['flesch_reading_ease'] = round(max(0, min(100, flesch)), 2)
        
        # 2. Flesch-Kincaid Grade Level
        # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        if word_count > 0 and sentence_count > 0:
            fk_grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
            scores['flesch_kincaid_grade'] = round(max(0, min(20, fk_grade)), 2)
        
        # 3. Gunning Fog Index
        # 0.4 * [(words/sentences) + 100 * (complex_words/words)]
        if word_count > 0 and sentence_count > 0:
            fog = 0.4 * ((word_count / sentence_count) + 100 * (complex_words / word_count))
            scores['gunning_fog'] = round(max(0, min(20, fog)), 2)
        
        # 4. SMOG Index
        # 1.0430 * sqrt(30 * (complex_words/sentences)) + 3.1291
        if sentence_count >= 3:
            smog = 1.0430 * math.sqrt(30 * (complex_words / sentence_count)) + 3.1291
            scores['smog_index'] = round(max(0, min(20, smog)), 2)
        else:
            scores['smog_index'] = scores.get('gunning_fog', 0)
        
        # 5. Coleman-Liau Index
        # 0.0588 * L - 0.296 * S - 15.8
        # L = average characters per 100 words, S = average sentences per 100 words
        if word_count > 0:
            L = (characters / word_count) * 100
            S = (sentence_count / word_count) * 100
            coleman = 0.0588 * L - 0.296 * S - 15.8
            scores['coleman_liau'] = round(max(0, min(20, coleman)), 2)
        
        # 6. Automated Readability Index
        # 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
        if word_count > 0 and sentence_count > 0:
            ari = 4.71 * (characters / word_count) + 0.5 * (word_count / sentence_count) - 21.43
            scores['automated_readability'] = round(max(0, min(20, ari)), 2)
        
        # Add interpretations
        interpretations = {}
        for score_name, value in scores.items():
            if score_name in NLPUtils.READABILITY_SCORES:
                ranges = NLPUtils.READABILITY_SCORES[score_name]['ranges']
                for low, high, label in ranges:
                    if low <= value < high:
                        interpretations[score_name] = label
                        break
        
        return {
            'scores': scores,
            'interpretations': interpretations,
            'metrics': {
                'sentence_count': sentence_count,
                'word_count': word_count,
                'syllable_count': syllable_count,
                'complex_word_count': complex_words,
                'char_count': characters
            }
        }
    
    @staticmethod
    def get_sentences(text: str) -> List[str]:
        """
        Get sentences from text
        Args:
            text: Input text
        Returns:
            List of sentences
        """
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback: simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def get_words(text: str, lowercase: bool = True) -> List[str]:
        """
        Get words from text
        Args:
            text: Input text
            lowercase: Convert to lowercase
        Returns:
            List of words
        """
        if NLTK_AVAILABLE:
            try:
                words = word_tokenize(text)
                words = [w for w in words if w.isalnum()]
                if lowercase:
                    words = [w.lower() for w in words]
                return words
            except:
                pass
        
        # Fallback: simple word splitting
        words = re.findall(r'\b\w+\b', text)
        if lowercase:
            words = [w.lower() for w in words]
        return words
    
    @staticmethod
    def count_syllables(text: str) -> int:
        """
        Count syllables in text
        Args:
            text: Input text
        Returns:
            Number of syllables
        """
        words = NLPUtils.get_words(text, lowercase=True)
        total = 0
        
        for word in words:
            # Count vowel groups as syllables
            syllable_count = len(re.findall(r'[aeiouy]+', word))
            
            # Adjust for silent e
            if word.endswith('e') and len(word) > 2:
                syllable_count = max(1, syllable_count - 1)
            
            # Ensure at least 1 syllable per word
            total += max(1, syllable_count)
        
        return total
    
    @staticmethod
    def count_complex_words(text: str, syllable_threshold: int = 3) -> int:
        """
        Count words with more than threshold syllables
        Args:
            text: Input text
            syllable_threshold: Minimum syllables to be considered complex
        Returns:
            Number of complex words
        """
        words = NLPUtils.get_words(text, lowercase=True)
        complex_count = 0
        
        for word in words:
            syllables = NLPUtils.count_syllables(word)
            if syllables >= syllable_threshold:
                complex_count += 1
        
        return complex_count
    
    @staticmethod
    def get_pos_tags(text: str) -> Dict[str, int]:
        """
        Get part of speech tag counts (simplified)
        Args:
            text: Input text
        Returns:
            Dictionary of POS tag counts
        """
        words = NLPUtils.get_words(text)
        pos_counts = {
            'noun': 0, 'verb': 0, 'adjective': 0, 'adverb': 0,
            'pronoun': 0, 'preposition': 0, 'conjunction': 0,
            'determiner': 0, 'number': 0, 'interjection': 0,
            'other': 0
        }
        
        if NLTK_AVAILABLE:
            try:
                from nltk import pos_tag
                tagged = pos_tag(words)
                
                for word, tag in tagged:
                    # Get simplified POS
                    simplified = NLPUtils._simplify_pos_tag(tag)
                    pos_counts[simplified] = pos_counts.get(simplified, 0) + 1
                
                return pos_counts
            except:
                pass
        
        # Fallback: simple heuristics
        for word in words:
            word_lower = word.lower()
            
            # Very basic POS guessing
            if word_lower in ['the', 'a', 'an']:
                pos_counts['determiner'] += 1
            elif word_lower in ['and', 'or', 'but', 'because', 'if']:
                pos_counts['conjunction'] += 1
            elif word_lower in ['in', 'on', 'at', 'by', 'for', 'with', 'about']:
                pos_counts['preposition'] += 1
            elif word_lower in ['i', 'you', 'he', 'she', 'it', 'we', 'they',
                               'me', 'him', 'her', 'us', 'them']:
                pos_counts['pronoun'] += 1
            elif word.isdigit():
                pos_counts['number'] += 1
            elif word_lower.endswith('ly'):
                pos_counts['adverb'] += 1
            elif word_lower.endswith(('ing', 'ed', 'ate')):
                pos_counts['verb'] += 1
            elif word_lower.endswith(('tion', 'ness', 'ment', 'ity')):
                pos_counts['noun'] += 1
            elif word_lower.endswith(('ous', 'ive', 'ful', 'less')):
                pos_counts['adjective'] += 1
            else:
                pos_counts['other'] += 1
        
        return pos_counts
    
    @staticmethod
    def _simplify_pos_tag(tag: str) -> str:
        """
        Simplify NLTK POS tag to basic categories
        Args:
            tag: NLTK POS tag
        Returns:
            Simplified POS category
        """
        # Extract base tag (remove tense markers)
        base = tag[:2] if tag else ''
        
        # Map to simplified categories
        if base in ['NN', 'NNS', 'NNP', 'NNPS']:
            return 'noun'
        elif base in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return 'verb'
        elif base in ['JJ', 'JJR', 'JJS']:
            return 'adjective'
        elif base in ['RB', 'RBR', 'RBS']:
            return 'adverb'
        elif base == 'PRP' or base == 'PRP$':
            return 'pronoun'
        elif base == 'DT':
            return 'determiner'
        elif base == 'IN':
            return 'preposition'
        elif base == 'CC':
            return 'conjunction'
        elif base == 'CD':
            return 'number'
        elif base == 'UH':
            return 'interjection'
        else:
            return 'other'
    
    @staticmethod
    def get_stopwords(language: str = 'english') -> set:
        """
        Get stopwords for specified language
        Args:
            language: Language code
        Returns:
            Set of stopwords
        """
        if NLTK_AVAILABLE:
            try:
                return set(stopwords.words(language))
            except:
                pass
        
        # Fallback: common English stopwords
        return {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'to', 'from', 'up',
            'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
            'will', 'just', 'don', 'should', 'now'
        }
    
    @staticmethod
    def stem_word(word: str) -> str:
        """
        Stem a word using Porter Stemmer
        Args:
            word: Input word
        Returns:
            Stemmed word
        """
        if NLTK_AVAILABLE:
            try:
                stemmer = PorterStemmer()
                return stemmer.stem(word)
            except:
                pass
        
        # Fallback: simple stemming
        if word.endswith('ing'):
            return word[:-3]
        elif word.endswith('ed'):
            return word[:-2]
        elif word.endswith('ly'):
            return word[:-2]
        elif word.endswith('s') and len(word) > 3:
            return word[:-1]
        else:
            return word
    
    @staticmethod
    def lemmatize_word(word: str, pos: str = 'n') -> str:
        """
        Lemmatize a word using WordNet
        Args:
            word: Input word
            pos: Part of speech (n, v, a, r)
        Returns:
            Lemmatized word
        """
        if NLTK_AVAILABLE:
            try:
                lemmatizer = WordNetLemmatizer()
                return lemmatizer.lemmatize(word, pos)
            except:
                pass
        
        return word
    
    @staticmethod
    def get_ngrams(text: str, n: int = 2, use_stemming: bool = False) -> List[str]:
        """
        Get n-grams from text
        Args:
            text: Input text
            n: N-gram size
            use_stemming: Apply stemming to words
        Returns:
            List of n-grams
        """
        words = NLPUtils.get_words(text, lowercase=True)
        
        if use_stemming:
            words = [NLPUtils.stem_word(w) for w in words]
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    @staticmethod
    def get_word_ngrams(text: str, n_range: Tuple[int, int] = (1, 3)) -> Dict[str, int]:
        """
        Get n-gram frequencies for a range of n values
        Args:
            text: Input text
            n_range: Range of n values (min, max)
        Returns:
            Dictionary of n-gram frequencies
        """
        result = {}
        
        for n in range(n_range[0], n_range[1] + 1):
            ngrams = NLPUtils.get_ngrams(text, n)
            counts = Counter(ngrams)
            result[f'{n}_grams'] = dict(counts.most_common(20))
        
        return result
    
    @staticmethod
    def get_lexical_diversity(text: str) -> Dict[str, float]:
        """
        Calculate various lexical diversity metrics
        Args:
            text: Input text
        Returns:
            Dictionary of diversity metrics
        """
        words = NLPUtils.get_words(text, lowercase=True)
        
        if not words:
            return {'error': 'No words to analyze'}
        
        unique_words = set(words)
        
        # Type-Token Ratio
        ttr = len(unique_words) / len(words)
        
        # Corrected TTR (for text length)
        cttr = len(unique_words) / math.sqrt(2 * len(words))
        
        # Hapax Legomena (words appearing once)
        word_counts = Counter(words)
        hapax_count = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax_count / len(words)
        
        # Hapax Dislegomena (words appearing twice)
        dis_count = sum(1 for count in word_counts.values() if count == 2)
        dis_ratio = dis_count / len(words)
        
        # Brunet's Index (vocabulary richness)
        try:
            brunet = len(words) ** (len(unique_words) ** -0.17)
        except:
            brunet = 0
        
        # Honoré's Statistic
        try:
            if hapax_count > 0:
                v1 = hapax_count
                v = len(unique_words)
                N = len(words)
                honore = 100 * math.log(N) / (1 - (v1 / v))
            else:
                honore = 0
        except:
            honore = 0
        
        return {
            'type_token_ratio': round(ttr, 3),
            'corrected_ttr': round(cttr, 3),
            'hapax_legomena_ratio': round(hapax_ratio, 3),
            'hapax_dislegomena_ratio': round(dis_ratio, 3),
            'brunet_index': round(brunet, 2),
            'honore_statistic': round(honore, 2),
            'unique_word_count': len(unique_words),
            'total_word_count': len(words)
        }
    
    @staticmethod
    def get_sentiment_scores(text: str) -> Dict[str, float]:
        """
        Simple sentiment analysis based on word lists
        Args:
            text: Input text
        Returns:
            Dictionary with sentiment scores
        """
        # Simple positive/negative word lists
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'beautiful', 'love', 'happy', 'glad', 'pleased', 'delighted',
            'perfect', 'best', 'awesome', 'brilliant', 'superb', 'outstanding',
            'positive', 'optimistic', 'hopeful', 'joy', 'joyful', 'enjoy'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'poor',
            'hate', 'angry', 'sad', 'upset', 'disappointed', 'frustrated',
            'annoying', 'awful', 'dreadful', 'negative', 'pessimistic',
            'fear', 'afraid', 'scared', 'worry', 'worried', 'anxious'
        }
        
        words = NLPUtils.get_words(text, lowercase=True)
        
        if not words:
            return {'positive': 0, 'negative': 0, 'neutral': 1.0}
        
        # Count positive and negative words
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        total_sentiment = pos_count + neg_count
        
        if total_sentiment == 0:
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        # Calculate ratios
        pos_ratio = pos_count / len(words)
        neg_ratio = neg_count / len(words)
        neutral_ratio = 1 - (pos_ratio + neg_ratio)
        
        # Compound score (-1 to 1)
        if pos_count > 0 or neg_count > 0:
            compound = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            compound = 0
        
        return {
            'positive': round(pos_ratio, 3),
            'negative': round(neg_ratio, 3),
            'neutral': round(neutral_ratio, 3),
            'compound': round(compound, 3)
        }
    
    @staticmethod
    def get_keyword_density(text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Calculate keyword density
        Args:
            text: Input text
            top_n: Number of top keywords to return
        Returns:
            List of keywords with frequencies
        """
        words = NLPUtils.get_words(text, lowercase=True)
        stopwords = NLPUtils.get_stopwords()
        
        # Remove stopwords
        content_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        if not content_words:
            return []
        
        # Count frequencies
        word_counts = Counter(content_words)
        total_words = len(content_words)
        
        # Calculate densities
        keywords = []
        for word, count in word_counts.most_common(top_n):
            density = (count / total_words) * 100
            keywords.append({
                'word': word,
                'count': count,
                'density': round(density, 2)
            })
        
        return keywords
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect language of text (simplified)
        Args:
            text: Input text
        Returns:
            Language code
        """
        # Check for common words in different languages
        language_markers = {
            'en': {'the', 'and', 'of', 'to', 'in', 'that', 'is', 'was'},
            'es': {'el', 'la', 'los', 'las', 'y', 'en', 'de', 'que'},
            'fr': {'le', 'la', 'les', 'et', 'en', 'de', 'que', 'est'},
            'de': {'der', 'die', 'das', 'und', 'in', 'von', 'mit', 'ist'},
            'it': {'il', 'la', 'i', 'gli', 'le', 'e', 'in', 'di', 'che'},
            'pt': {'o', 'a', 'os', 'as', 'e', 'em', 'de', 'que'},
            'nl': {'de', 'het', 'een', 'en', 'in', 'van', 'met', 'is'},
            'ru': {'и', 'в', 'не', 'на', 'я', 'быть', 'он', 'с'},
            'zh': {'的', '了', '和', '是', '就', '都', '而', '及'},
            'ja': {'の', 'に', 'を', 'は', 'が', 'で', 'と', 'も'}
        }
        
        words = set(NLPUtils.get_words(text, lowercase=True))
        
        # Score each language
        scores = {}
        for lang, markers in language_markers.items():
            matches = words.intersection(markers)
            scores[lang] = len(matches)
        
        if not scores:
            return 'unknown'
        
        # Get language with highest score
        detected = max(scores, key=scores.get)
        max_score = scores[detected]
        
        # If no significant matches, return unknown
        if max_score == 0:
            return 'unknown'
        
        return detected
    
    @staticmethod
    def calculate_entropy(text: str) -> Dict[str, float]:
        """
        Calculate Shannon entropy of text
        Args:
            text: Input text
        Returns:
            Dictionary with entropy scores
        """
        if not text:
            return {'character_entropy': 0, 'word_entropy': 0, 'combined': 0}
        
        # Character-level entropy
        char_counts = Counter(text)
        total_chars = len(text)
        
        char_entropy = 0
        for count in char_counts.values():
            prob = count / total_chars
            char_entropy -= prob * math.log2(prob)
        
        # Word-level entropy
        words = NLPUtils.get_words(text, lowercase=True)
        if words:
            word_counts = Counter(words)
            total_words = len(words)
            
            word_entropy = 0
            for count in word_counts.values():
                prob = count / total_words
                word_entropy -= prob * math.log2(prob)
        else:
            word_entropy = 0
        
        # Normalize to 0-100 scale
        # Typical character entropy: 4-8 bits
        # Typical word entropy: 8-12 bits
        
        char_normalized = min(100, max(0, (char_entropy - 4) * 25))
        word_normalized = min(100, max(0, (word_entropy - 8) * 25))
        
        combined = (char_normalized + word_normalized) / 2
        
        return {
            'character_entropy': round(char_entropy, 2),
            'character_entropy_normalized': round(char_normalized, 2),
            'word_entropy': round(word_entropy, 2),
            'word_entropy_normalized': round(word_normalized, 2),
            'combined_normalized': round(combined, 2)
        }
    
    @staticmethod
    def get_basic_stats(text: str) -> Dict[str, Any]:
        """
        Get basic text statistics
        Args:
            text: Input text
        Returns:
            Dictionary with basic stats
        """
        sentences = NLPUtils.get_sentences(text)
        words = NLPUtils.get_words(text)
        
        if not sentences or not words:
            return {
                'char_count': len(text),
                'word_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0
            }
        
        # Count paragraphs
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        # Calculate averages
        avg_word_length = sum(len(w) for w in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'unique_word_count': len(set(w.lower() for w in words)),
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'longest_word': max(words, key=len) if words else '',
            'shortest_word': min(words, key=len) if words else ''
        }