import re
from collections import Counter
import numpy as np
from typing import List, Tuple  # Add for type hints

class PatternAnalyzer:
    """
    Recognizes specific patterns in AI-generated text
    """
    
    def __init__(self):
        # Common AI phrases and patterns
        self.ai_phrases = [
            r'in recent years', r'it is important to note',
            r'there are several', r'one of the most',
            r'as we have seen', r'in conclusion', r'furthermore',
            r'moreover', r'additionally', r'consequently',
            r'as a result', r'for this reason', r'it should be noted',
            r'it is worth mentioning', r'it is clear that'
        ]
        
        self.ai_sentence_starters = [
            r'^in conclusion', r'^furthermore', r'^moreover',
            r'^additionally', r'^consequently', r'^therefore',
            r'^thus', r'^hence', r'^overall', r'^ultimately'
        ]
        
        # Human patterns (lack of these suggests AI)
        self.human_markers = [
            r'\bi\s+(think|believe|feel|guess|suppose)\b',
            r'\bmaybe\b', r'\bperhaps\b', r'\bprobably\b',
            r'\bkinda\b', r'\bsorta\b', r'\blike\s+when\b',
            r'\bi\s+remember\b', r'\bback\s+then\b',
            r'\bwhen\s+I\s+was\b'
        ]
        
        # Transition word patterns (AI overuses certain transitions)
        self.transition_chains = [
            (r'however', r'therefore'), (r'furthermore', r'moreover'),
            (r'in addition', r'additionally'), (r'first', r'second', r'third')
        ]
    
    def detect_ai_patterns(self, text):
        """
        Detect common AI writing patterns
        """
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 50.0
        
        # Count AI phrases
        ai_phrase_count = 0
        for pattern in self.ai_phrases:
            ai_phrase_count += len(re.findall(pattern, text_lower))
        
        ai_phrase_density = (ai_phrase_count / len(sentences)) * 100
        
        # Count sentence starters
        starter_count = 0
        for sentence in sentences[:5]:  # Check first 5 sentences
            sentence_lower = sentence.lower().strip()
            for pattern in self.ai_sentence_starters:
                if re.match(pattern, sentence_lower):
                    starter_count += 1
        
        # Check for transition chains
        chain_count = 0
        for chain in self.transition_chains:
            chain_found = all(
                re.search(pattern, text_lower) for pattern in chain
            )
            if chain_found:
                chain_count += 1
        
        # Score calculation
        # High AI phrase density = AI-like
        if ai_phrase_density > 30:
            ai_score = 20
        elif ai_phrase_density > 15:
            ai_score = 40
        else:
            ai_score = 70
        
        # Many sentence starters = AI-like
        if starter_count > 3:
            starter_score = 30
        elif starter_count > 1:
            starter_score = 50
        else:
            starter_score = 80
        
        # Transition chains = AI-like
        chain_score = max(0, 100 - (chain_count * 30))
        
        # Average all scores
        final_score = (ai_score + starter_score + chain_score) / 3
        
        return {
            'ai_phrase_density': round(ai_phrase_density, 2),
            'sentence_starter_count': starter_count,
            'transition_chains': chain_count,
            'pattern_score': round(final_score, 2)
        }
    
    def detect_human_patterns(self, text):
        """
        Detect patterns that suggest human writing
        """
        text_lower = text.lower()
        
        # Count human markers
        human_marker_count = 0
        for pattern in self.human_markers:
            human_marker_count += len(re.findall(pattern, text_lower))
        
        # Check for personal stories
        story_indicators = [
            r'last\s+(week|month|year|monday|tuesday)',
            r'yesterday', r'today', r'tomorrow',
            r'when\s+I', r'after\s+I', r'before\s+I'
        ]
        
        story_count = 0
        for indicator in story_indicators:
            story_count += len(re.findall(indicator, text_lower))
        
        # Check for informal language
        informal_words = [
            r'\bawesome\b', r'\bcool\b', r'\bamazing\b',
            r'\bterrible\b', r'\bhorrible\b', r'\bunbelievable\b',
            r'\bgot\b', r'\bgonna\b', r'\bwanna\b'
        ]
        
        informal_count = 0
        for word in informal_words:
            informal_count += len(re.findall(word, text_lower))
        
        # Calculate human score
        words = text_lower.split()
        word_count = len(words) if words else 1
        
        human_density = ((human_marker_count + story_count + informal_count) / word_count) * 1000
        
        if human_density > 20:
            return 85.0  # Strong human patterns
        elif human_density > 10:
            return 65.0  # Moderate human patterns
        elif human_density > 5:
            return 45.0  # Some human patterns
        else:
            return 25.0  # Few human patterns (AI-like)
        
    def analyze_repetition_patterns(self, text):
        """
        Detect repetitive patterns unique to AI
        """
        words = text.lower().split()
        if len(words) < 20:
            return 50.0
        
        # Check for repeated sentence structures
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 50.0
        
        # Get first 3 words of each sentence
        openings = []
        for sent in sentences:
            first_words = ' '.join(sent.split()[:3])
            openings.append(first_words.lower())
        
        # Check for repeated openings
        opening_counts = Counter(openings)
        repeated_openings = sum(1 for count in opening_counts.values() if count > 1)
        
        # Get sentence lengths
        lengths = [len(sent.split()) for sent in sentences]
        length_std = np.std(lengths) if len(lengths) > 1 else 0
        
        # AI often has:
        # - Repeated sentence openings
        # - Uniform sentence lengths (low standard deviation)
        
        if repeated_openings > len(sentences) * 0.3:  # 30% repetition
            return 25.0  # AI-like
        elif length_std < 3:  # Very uniform lengths
            return 30.0  # AI-like
        elif length_std > 8:  # Very varied
            return 80.0  # Human-like
        else:
            return 60.0  # Mixed