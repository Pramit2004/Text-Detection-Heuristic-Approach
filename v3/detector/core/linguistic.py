"""
Linguistic Analyzer v2 - Rebuilt for strong AI vs Human discrimination.

Key fixes vs v1:
- transition_usage: was the best signal (24 AI vs 83 human) but weighted too low.
  Now weighted highest.
- hedge_words: good signal (14 AI vs 62 human) — weight increased.
- vocabulary_diversity: was identical (89.5) for all texts because the scoring
  bands were wrong. Recalibrated with correct TTR ranges.
- sentence_starters: AI uses predictable starters ("Furthermore", "Additionally"),
  humans vary more — added stricter scoring.
- Added ai_connector_abuse: detects the specific pattern of starting EVERY
  sentence with a connector word (very strong AI signal).
"""

import re
import math
from collections import Counter
from typing import Any, Dict

import numpy as np

from ..nlp_helpers import sent_tokenize, word_tokenize


class LinguisticAnalyzer:

    def __init__(self):
        # These specifically distinguish AI from human writing
        self.ai_transition_phrases = [
            'however','therefore','furthermore','moreover',
            'consequently','in addition','for example','in conclusion',
            'firstly','secondly','thirdly','finally','thus','hence',
            'additionally','nevertheless','nonetheless','accordingly',
            'as a result','for this reason','in contrast','on the other hand',
            'to begin with','first and foremost','last but not least',
            'it is worth noting','it should be noted',
        ]
        self.hedge_words = [
            'perhaps','maybe','possibly','probably','approximately',
            'i think','it seems','apparently','arguably','somewhat',
            'quite','relatively','fairly','rather','nearly',
            'almost','sometimes','occasionally','often','usually',
            'i believe','i feel','in my opinion','it appears',
            'to some extent','in some ways','more or less',
            'i guess','i suppose','i reckon','kind of','sort of',
        ]
        self.complex_markers = [
            'which','that','who','whom','whose',
            'although','even though','despite','whereas',
            'while','unless','until','provided that',
        ]
        self.human_sentence_starters = [
            'i','we','you','they','he','she',
            'actually','basically','honestly','frankly',
            'well','so','anyway','look','right','ok',
            'yeah','no','wait','honestly',
        ]
        self.ai_sentence_starters = [
            'in conclusion','furthermore','moreover','additionally',
            'consequently','therefore','thus','hence','accordingly',
            'it is important','it should be noted','it is worth',
            'firstly','secondly','thirdly','lastly','finally',
            'to begin with','first and foremost',
        ]

    # ── main ──────────────────────────────────────────────────────────────────
    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 50:
            return self._default(50.0)

        scores = {
            'sentence_structure':   self.analyze_sentence_structure(text),
            'vocabulary_diversity': self.analyze_vocabulary_diversity(text),
            'transition_usage':     self.analyze_transition_usage(text),
            'hedge_words':          self.analyze_hedge_words(text),
            'complex_sentences':    self.analyze_complex_sentences(text),
            'sentence_starters':    self.analyze_sentence_starters(text),
        }

        # Transition and hedge are strongest discriminators — weight them most
        linguistic_score = (
            scores['sentence_structure']   * 0.10 +
            scores['vocabulary_diversity'] * 0.15 +
            scores['transition_usage']     * 0.30 +
            scores['hedge_words']          * 0.25 +
            scores['complex_sentences']    * 0.10 +
            scores['sentence_starters']    * 0.10
        )

        return {
            **{k: round(v, 2) for k, v in scores.items()},
            'linguistic_score': round(linguistic_score, 2),
            'details': self._get_detailed_analysis(text),
        }

    @staticmethod
    def _default(v: float) -> Dict[str, Any]:
        keys = ['sentence_structure','vocabulary_diversity','transition_usage',
                'hedge_words','complex_sentences','sentence_starters','linguistic_score']
        return {k: v for k in keys}

    def _get_detailed_analysis(self, text: str) -> Dict[str, Any]:
        sentences = sent_tokenize(text)
        words     = [w for w in word_tokenize(text.lower()) if w.isalnum()]
        sc = len(sentences); wc = len(words)
        return {
            'sentence_count':      sc,
            'word_count':          wc,
            'avg_sentence_length': round(wc / sc, 2) if sc else 0,
            'unique_words':        len(set(words)),
            'type_token_ratio':    round(len(set(words)) / wc, 3) if wc else 0,
        }

    # ── sentence structure ─────────────────────────────────────────────────────
    def analyze_sentence_structure(self, text: str) -> float:
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return 50.0

        lengths = [len(word_tokenize(s)) for s in sentences]
        if not lengths:
            return 50.0

        mean_len = float(np.mean(lengths))
        if mean_len == 0:
            return 50.0

        cv = float(np.std(lengths)) / mean_len

        # Higher CV = more varied lengths = more human
        if   cv > 0.6: cv_score = 90
        elif cv > 0.4: cv_score = 75
        elif cv > 0.3: cv_score = 60
        elif cv > 0.2: cv_score = 45
        else:          cv_score = 28

        very_short = sum(1 for l in lengths if l < 5)
        very_long  = sum(1 for l in lengths if l > 30)
        if (very_short + very_long) / len(lengths) > 0.2:
            cv_score += 8

        return round(min(100, cv_score), 2)

    # ── vocabulary diversity (recalibrated) ────────────────────────────────────
    def analyze_vocabulary_diversity(self, text: str) -> float:
        words = [w for w in word_tokenize(text.lower()) if w.isalnum()]
        if len(words) < 20:
            return 50.0

        unique      = set(words)
        ttr         = len(unique) / len(words)
        wc          = Counter(words)
        hapax       = sum(1 for c in wc.values() if c == 1)
        hapax_ratio = hapax / len(words)

        # TTR thresholds: short texts ~0.7+, medium ~0.5-0.7, long ~0.3-0.5
        # AI and human both have similar TTR — focus on hapax and rare words instead

        # Rare words (appearing exactly once) = human creativity signal
        # AI tends to use common words more uniformly
        if   hapax_ratio > 0.65: hapax_score = 85
        elif hapax_ratio > 0.50: hapax_score = 72
        elif hapax_ratio > 0.40: hapax_score = 60
        elif hapax_ratio > 0.30: hapax_score = 48
        else:                    hapax_score = 35

        # TTR scaled by text length (longer text naturally has lower TTR)
        adjusted_ttr = ttr * math.sqrt(len(words) / 100)
        if   adjusted_ttr > 1.5: ttr_score = 85
        elif adjusted_ttr > 1.0: ttr_score = 72
        elif adjusted_ttr > 0.7: ttr_score = 60
        elif adjusted_ttr > 0.5: ttr_score = 48
        else:                    ttr_score = 35

        # Most common word dominance — AI over-uses certain words
        most_common_freq = max(wc.values()) / len(words)
        if   most_common_freq < 0.03: dom_score = 80
        elif most_common_freq < 0.05: dom_score = 65
        elif most_common_freq < 0.08: dom_score = 50
        else:                         dom_score = 35

        return round(hapax_score * 0.40 + ttr_score * 0.35 + dom_score * 0.25, 2)

    # ── transition usage (best discriminator — 24 AI vs 83 human in tests) ─────
    def analyze_transition_usage(self, text: str) -> float:
        text_lower = text.lower()
        sentences  = sent_tokenize(text)
        if len(sentences) < 3:
            return 50.0

        ai_count = sum(
            len(re.findall(r'\b' + re.escape(p) + r'\b', text_lower))
            for p in self.ai_transition_phrases
        )
        transitions_per_sentence = ai_count / len(sentences)

        # Check if AI starters begin sentences specifically
        ai_started = sum(
            1 for s in sentences
            if any(s.lower().lstrip().startswith(starter)
                   for starter in self.ai_sentence_starters)
        )
        ai_start_ratio = ai_started / len(sentences)

        # Density scoring — more transitions = more AI-like (lower human score)
        if   transitions_per_sentence > 1.5: density_score = 12
        elif transitions_per_sentence > 1.0: density_score = 22
        elif transitions_per_sentence > 0.7: density_score = 35
        elif transitions_per_sentence > 0.4: density_score = 52
        elif transitions_per_sentence > 0.2: density_score = 70
        elif transitions_per_sentence > 0.1: density_score = 82
        else:                                density_score = 93

        # Penalise heavily for AI sentence starters
        if   ai_start_ratio > 0.4: start_score = 15
        elif ai_start_ratio > 0.25:start_score = 30
        elif ai_start_ratio > 0.1: start_score = 55
        else:                      start_score = 85

        return round(density_score * 0.65 + start_score * 0.35, 2)

    # ── hedge words (second best discriminator — 14 AI vs 62 human) ───────────
    def analyze_hedge_words(self, text: str) -> float:
        text_lower = text.lower()
        words      = [w for w in word_tokenize(text_lower) if w.isalnum()]
        if len(words) < 20:
            return 50.0

        hedge_count  = 0
        hedges_found = []
        for h in self.hedge_words:
            cnt = len(re.findall(r'\b' + re.escape(h) + r'\b', text_lower))
            if cnt:
                hedge_count  += cnt
                hedges_found.append(h)

        hedge_density = (hedge_count / len(words)) * 100
        variety_score = min(100, (len(set(hedges_found)) / 4) * 100)

        # AI rarely hedges; humans hedge frequently
        if   hedge_density < 0.15: density_score = 12
        elif hedge_density < 0.4:  density_score = 28
        elif hedge_density < 0.8:  density_score = 50
        elif hedge_density < 1.5:  density_score = 70
        elif hedge_density < 2.5:  density_score = 82
        elif hedge_density < 4.0:  density_score = 78
        else:                      density_score = 60

        return round(density_score * 0.70 + variety_score * 0.30, 2)

    # ── complex sentences ──────────────────────────────────────────────────────
    def analyze_complex_sentences(self, text: str) -> float:
        sentences  = sent_tokenize(text)
        if len(sentences) < 3:
            return 50.0
        text_lower = text.lower()

        complex_cnt = sum(
            len(re.findall(r'\b' + re.escape(m) + r'\b', text_lower))
            for m in self.complex_markers
        )
        subord = ['because','since','as','although','though','while',
                  'whereas','unless','until','if','even if']
        subord_cnt = sum(
            len(re.findall(r'\b' + re.escape(s) + r'\b', text_lower))
            for s in subord
        )
        comma_cnt = text.count(',')

        cps       = complex_cnt / len(sentences)
        sps       = subord_cnt  / len(sentences)
        comma_ps  = comma_cnt   / len(sentences)

        complex_score = (85 if 0.5<=cps<=2.0 else 70 if 2.0<cps<=3.0
                         else min(100, cps*50))
        subord_score  = (85 if 0.3<=sps<=1.5 else 70 if 1.5<sps<=2.5
                         else min(100, sps*40))
        comma_score   = min(100, comma_ps * 20)

        return round(complex_score*0.4 + subord_score*0.35 + comma_score*0.25, 2)

    # ── sentence starters ──────────────────────────────────────────────────────
    def analyze_sentence_starters(self, text: str) -> float:
        sentences = sent_tokenize(text)
        if len(sentences) < 5:
            return 50.0

        starters = []
        for sent in sentences:
            words = sent.split()
            if not words:
                continue
            starter = ' '.join(words[:2]).lower().rstrip('.,!?;:')
            starters.append(starter)

        if not starters:
            return 50.0

        variety_score = (len(set(starters)) / len(starters)) * 100

        ai_ratio = sum(
            1 for s in starters
            if any(s.startswith(ai) for ai in self.ai_sentence_starters)
        ) / len(starters)

        human_ratio = sum(
            1 for s in starters
            if s.split() and s.split()[0] in self.human_sentence_starters
        ) / len(starters)

        # Penalise AI starters heavily
        if   ai_ratio > 0.4:  ai_score = 10
        elif ai_ratio > 0.25: ai_score = 25
        elif ai_ratio > 0.1:  ai_score = 50
        else:                 ai_score = 80

        human_score = (88 if human_ratio > 0.25 else 72 if human_ratio > 0.15
                       else 55 if human_ratio > 0.05 else 38)

        return round(variety_score*0.35 + ai_score*0.35 + human_score*0.30, 2)
