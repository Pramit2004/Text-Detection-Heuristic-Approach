"""
Text Processing Utilities - Complete implementation
Provides text cleaning, normalization, and extraction functions
"""

import re
import unicodedata
import html
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import math
import string

class TextProcessor:
    """
    Utility functions for text processing, cleaning, and normalization
    """
    
    # Common contractions and their expansions
    CONTRACTIONS = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "might've": "might have",
        "mightn't": "might not",
        "must've": "must have",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what's": "what is",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who's": "who is",
        "won't": "will not",
        "would've": "would have",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    
    # Common abbreviations
    ABBREVIATIONS = {
        "mr.": "mister",
        "mrs.": "missus",
        "ms.": "ms",
        "dr.": "doctor",
        "prof.": "professor",
        "rev.": "reverend",
        "hon.": "honorable",
        "pres.": "president",
        "gov.": "governor",
        "sen.": "senator",
        "rep.": "representative",
        "st.": "saint",
        "jr.": "junior",
        "sr.": "senior",
        "ph.d.": "phd",
        "m.d.": "md",
        "b.a.": "ba",
        "b.s.": "bs",
        "m.a.": "ma",
        "m.s.": "ms",
        "etc.": "etcetera",
        "e.g.": "for example",
        "i.e.": "that is",
        "vs.": "versus",
        "inc.": "incorporated",
        "corp.": "corporation",
        "ltd.": "limited",
        "co.": "company"
    }
    
    # Common stop words
    STOP_WORDS = set([
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn',
        'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
        'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
    ])
    
    @staticmethod
    def clean_text(text: str, options: Dict[str, bool] = None) -> str:
        """
        Comprehensive text cleaning
        Args:
            text: Input text
            options: Cleaning options dictionary
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Default options
        default_options = {
            'normalize_unicode': True,
            'remove_html': True,
            'expand_contractions': True,
            'fix_abbreviations': True,
            'remove_extra_spaces': True,
            'remove_extra_newlines': True,
            'normalize_quotes': True,
            'remove_control_chars': True,
            'strip_whitespace': True
        }
        
        if options:
            default_options.update(options)
        
        text = str(text)
        
        # Normalize unicode
        if default_options['normalize_unicode']:
            text = unicodedata.normalize('NFKD', text)
        
        # Remove HTML tags
        if default_options['remove_html']:
            text = re.sub(r'<[^>]+>', ' ', text)
            text = html.unescape(text)
        
        # Remove control characters
        if default_options['remove_control_chars']:
            text = ''.join(char for char in text if ord(char) >= 32 or char == '\n' or char == '\t')
        
        # Expand contractions
        if default_options['expand_contractions']:
            text = TextProcessor.expand_contractions(text)
        
        # Fix abbreviations
        if default_options['fix_abbreviations']:
            text = TextProcessor.fix_abbreviations(text)
        
        # Normalize quotes
        if default_options['normalize_quotes']:
            text = TextProcessor.normalize_quotes(text)
        
        # Remove extra newlines
        if default_options['remove_extra_newlines']:
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove extra spaces
        if default_options['remove_extra_spaces']:
            text = re.sub(r' +', ' ', text)
        
        # Strip whitespace
        if default_options['strip_whitespace']:
            text = text.strip()
        
        return text
    
    @staticmethod
    def expand_contractions(text: str) -> str:
        """
        Expand common contractions
        Args:
            text: Input text
        Returns:
            Text with contractions expanded
        """
        text_lower = text.lower()
        words = text.split()
        expanded_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in TextProcessor.CONTRACTIONS:
                # Preserve original case pattern if possible
                expansion = TextProcessor.CONTRACTIONS[word_lower]
                if word[0].isupper():
                    expansion = expansion.capitalize()
                expanded_words.append(expansion)
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    @staticmethod
    def fix_abbreviations(text: str) -> str:
        """
        Expand common abbreviations
        Args:
            text: Input text
        Returns:
            Text with abbreviations expanded
        """
        words = text.split()
        fixed_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in TextProcessor.ABBREVIATIONS:
                expansion = TextProcessor.ABBREVIATIONS[word_lower]
                if word[0].isupper():
                    expansion = expansion.capitalize()
                fixed_words.append(expansion)
            else:
                fixed_words.append(word)
        
        return ' '.join(fixed_words)
    
    @staticmethod
    def normalize_quotes(text: str) -> str:
        """
        Normalize various quote styles to standard quotes
        Args:
            text: Input text
        Returns:
            Text with normalized quotes
        """
        # Replace fancy quotes with straight quotes
        quote_replacements = [
            ('"', '"'), ('"', '"'),  # Smart quotes
            (''', "'"), (''', "'"),   # Smart apostrophes
            ('«', '"'), ('»', '"'),   # Guillemets
            ('„', '"'), ('“', '"'),   # Low double quotes
            ('‘', "'"), ('’', "'"),   # Low single quotes
            ('‚', "'"), ('‛', "'"),   # Low quotes
        ]
        
        for fancy, straight in quote_replacements:
            text = text.replace(fancy, straight)
        
        return text
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences with improved handling
        Args:
            text: Input text
        Returns:
            List of sentences
        """
        # Handle common abbreviations that shouldn't split sentences
        abbrev_pattern = r'\b(?:{})\b'.format('|'.join([
            'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Rev', 'Hon', 'St',
            'Jr', 'Sr', 'Gen', 'Col', 'Maj', 'Capt', 'Lt', 'Sgt',
            'vs', 'etc', 'e.g', 'i.e', 'al'
        ]))
        
        # Temporarily replace abbreviations with placeholder
        placeholders = {}
        def replace_abbrev(match):
            placeholder = f"__ABBREV_{len(placeholders)}__"
            placeholders[placeholder] = match.group(0)
            return placeholder
        
        text = re.sub(abbrev_pattern, replace_abbrev, text, flags=re.IGNORECASE)
        
        # Split on sentence boundaries
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, text)
        
        # Restore abbreviations
        result = []
        for sentence in sentences:
            for placeholder, original in placeholders.items():
                sentence = sentence.replace(placeholder, original)
            result.append(sentence.strip())
        
        return [s for s in result if s]
    
    @staticmethod
    def split_into_paragraphs(text: str) -> List[str]:
        """
        Split text into paragraphs
        Args:
            text: Input text
        Returns:
            List of paragraphs
        """
        # Split on double newlines or multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    @staticmethod
    def split_into_words(text: str, remove_punctuation: bool = True) -> List[str]:
        """
        Split text into words
        Args:
            text: Input text
            remove_punctuation: Whether to remove punctuation
        Returns:
            List of words
        """
        if remove_punctuation:
            # Remove punctuation but keep apostrophes in contractions
            text = re.sub(r'[^\w\s\']', ' ', text)
        
        words = text.split()
        return [w for w in words if w]
    
    @staticmethod
    def get_word_frequencies(text: str, ignore_stopwords: bool = True) -> Dict[str, int]:
        """
        Get word frequency dictionary
        Args:
            text: Input text
            ignore_stopwords: Whether to ignore common stop words
        Returns:
            Dictionary of word frequencies
        """
        words = TextProcessor.split_into_words(text.lower())
        
        if ignore_stopwords:
            words = [w for w in words if w not in TextProcessor.STOP_WORDS]
        
        return dict(Counter(words))
    
    @staticmethod
    def get_ngrams(text: str, n: int = 2, ignore_stopwords: bool = True) -> List[str]:
        """
        Extract n-grams from text
        Args:
            text: Input text
            n: N-gram size (2 for bigrams, 3 for trigrams, etc.)
            ignore_stopwords: Whether to ignore stop words
        Returns:
            List of n-grams
        """
        words = TextProcessor.split_into_words(text.lower())
        
        if ignore_stopwords:
            words = [w for w in words if w not in TextProcessor.STOP_WORDS]
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    @staticmethod
    def extract_email_addresses(text: str) -> List[str]:
        """
        Extract email addresses from text
        Args:
            text: Input text
        Returns:
            List of email addresses
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """
        Extract URLs from text
        Args:
            text: Input text
        Returns:
            List of URLs
        """
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:[/\w\.-]*)*(?:\?[\w=&]*)?(?:#[\w-]*)?'
        return re.findall(url_pattern, text)
    
    @staticmethod
    def extract_phone_numbers(text: str) -> List[str]:
        """
        Extract phone numbers from text
        Args:
            text: Input text
        Returns:
            List of phone numbers
        """
        phone_patterns = [
            r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
            r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',  # Another US format
        ]
        
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        
        return list(set(phones))
    
    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """
        Extract dates from text
        Args:
            text: Input text
        Returns:
            List of dates
        """
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',           # MM/DD/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',                  # YYYY-MM-DD
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',            # MM-DD-YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return list(set(dates))
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """
        Remove extra whitespace from text
        Args:
            text: Input text
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Replace multiple tabs with single tab
        text = re.sub(r'\t+', '\t', text)
        return text.strip()
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 1000, ellipsis: str = "...") -> str:
        """
        Truncate text to maximum length
        Args:
            text: Input text
            max_length: Maximum length
            ellipsis: Ellipsis string
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclamation = truncated.rfind('!')
        
        last_sentence = max(last_period, last_question, last_exclamation)
        
        if last_sentence > max_length * 0.8:  # If we found a sentence boundary
            return text[:last_sentence + 1] + " " + ellipsis
        else:
            # Truncate at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:
                return text[:last_space] + " " + ellipsis
            else:
                return text[:max_length] + ellipsis
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Simple language detection based on character sets
        Args:
            text: Input text
        Returns:
            Detected language code
        """
        # This is a simplified detection - for production, use langdetect or similar
        text_lower = text.lower()
        
        # Check for common English words
        english_words = ['the', 'and', 'of', 'to', 'in', 'that', 'is', 'was']
        english_count = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')
        
        if english_count > 3:
            return 'en'
        
        # Check for Spanish
        spanish_words = ['el', 'la', 'los', 'las', 'y', 'en', 'de', 'que']
        spanish_count = sum(1 for word in spanish_words if f' {word} ' in f' {text_lower} ')
        
        if spanish_count > 3:
            return 'es'
        
        # Check for French
        french_words = ['le', 'la', 'les', 'et', 'en', 'de', 'que', 'est']
        french_count = sum(1 for word in french_words if f' {word} ' in f' {text_lower} ')
        
        if french_count > 3:
            return 'fr'
        
        # Check for German
        german_words = ['der', 'die', 'das', 'und', 'in', 'von', 'mit', 'ist']
        german_count = sum(1 for word in german_words if f' {word} ' in f' {text_lower} ')
        
        if german_count > 3:
            return 'de'
        
        return 'unknown'
    
    @staticmethod
    def get_readability_stats(text: str) -> Dict[str, float]:
        """
        Calculate basic readability statistics
        Args:
            text: Input text
        Returns:
            Dictionary of readability metrics
        """
        sentences = TextProcessor.split_into_sentences(text)
        words = TextProcessor.split_into_words(text)
        characters = len(text)
        
        if not sentences or not words:
            return {
                'sentence_count': 0,
                'word_count': 0,
                'char_count': 0,
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'avg_chars_per_word': 0
            }
        
        # Basic statistics
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Average sentence length (in words)
        avg_sentence_length = word_count / sentence_count
        
        # Average word length (in characters)
        total_word_chars = sum(len(word) for word in words)
        avg_word_length = total_word_chars / word_count
        
        # Average characters per word (including spaces)
        avg_chars_per_word = characters / word_count
        
        return {
            'sentence_count': sentence_count,
            'word_count': word_count,
            'char_count': characters,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2),
            'avg_chars_per_word': round(avg_chars_per_word, 2)
        }
    
    @staticmethod
    def mask_pii(text: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Mask personally identifiable information
        Args:
            text: Input text
        Returns:
            Tuple of (masked_text, dict of masked items)
        """
        masked = text
        masked_items = {
            'emails': [],
            'phones': [],
            'names': [],  # Basic name detection
            'locations': []  # Basic location detection
        }
        
        # Mask emails
        emails = TextProcessor.extract_email_addresses(text)
        for i, email in enumerate(emails):
            masked = masked.replace(email, f'[EMAIL_{i+1}]')
            masked_items['emails'].append(email)
        
        # Mask phone numbers
        phones = TextProcessor.extract_phone_numbers(text)
        for i, phone in enumerate(phones):
            masked = masked.replace(phone, f'[PHONE_{i+1}]')
            masked_items['phones'].append(phone)
        
        # Simple name masking (capitalized words that aren't at start of sentence)
        # This is simplified - use NER for production
        words = text.split()
        for i, word in enumerate(words):
            if word and word[0].isupper() and i > 0:
                prev_word = words[i-1].lower()
                # Check if it might be a name
                if prev_word not in ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.']:
                    if word not in masked_items['names']:
                        masked_items['names'].append(word)
                        masked = masked.replace(word, f'[NAME_{len(masked_items["names"])}]')
        
        return masked, masked_items
    
    @staticmethod
    def normalize_for_comparison(text: str) -> str:
        """
        Normalize text for comparison (lowercase, remove punctuation, etc.)
        Args:
            text: Input text
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers (optional - depends on use case)
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts (0-100)
        Args:
            text1: First text
            text2: Second text
        Returns:
            Similarity score (0-100)
        """
        # Normalize texts
        norm1 = TextProcessor.normalize_for_comparison(text1)
        norm2 = TextProcessor.normalize_for_comparison(text2)
        
        # Get word sets
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union)
        
        return round(jaccard * 100, 2)