"""
Statistical Analyzer v2 - Rebuilt for strong AI vs Human discrimination.

Key fixes vs v1:
- repetition: was giving 92-93 to BOTH AI and human (useless). Now measures
  structural bigram/trigram repetition which actually differs.
- vocabulary_diversity: was identical (89.5) for all texts. Recalibrated.
- Added AI phrase density: the strongest statistical signal (transition overuse).
- Added human marker density: personal pronouns + casual language.
- Recalibrated all scoring bands with empirical ranges.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List

import numpy as np

# ── strongest AI signal: overused filler/transition phrases ──────────────────
_AI_PHRASES = [
    "furthermore","moreover","additionally","consequently","therefore",
    "thus","hence","in conclusion","in summary","to summarize",
    "it is important to note","it is worth noting","it should be noted",
    "as previously mentioned","as previously stated","as mentioned above",
    "needless to say","it goes without saying","last but not least",
    "in today's world","in today's society","in today's fast-paced",
    "in the modern era","one of the most","there are several key",
    "it is clear that","it is evident that","it is undeniable",
    "plays a crucial role","plays a vital role","plays an important role",
    "a wide range of","a variety of factors","in order to",
    "it is essential","it is necessary","it is imperative",
    "delve into","dive deep","unlock the secrets","journey through",
    "at the end of the day","when it comes to","in terms of",
    "it is worth mentioning","on the other hand","needless to say",
    "taking into account","taking into consideration",
]

# ── human signals: informal, personal, spontaneous language ──────────────────
_HUMAN_PATTERNS = [
    r'\bi\b', r'\bme\b', r'\bmy\b', r'\bmine\b', r'\bmyself\b',
    r'\bwe\b', r'\bus\b', r'\bour\b',
    r'\bactually\b', r'\bhonestly\b', r'\bfunny\b', r'\bwait\b',
    r'\banyway\b', r'\bstuff\b', r'\bthing is\b', r'\bkinda\b',
    r'\bsorta\b', r'\bngl\b', r'\bidk\b', r'\btbh\b',
    r'\byeah\b', r'\bnope\b', r'\byep\b', r'\bhaha\b',
    r'\blol\b', r'\bomg\b', r'\bwtf\b', r'\bbtw\b',
    r"i'm\b", r"i've\b", r"i'd\b", r"i'll\b", r"can't\b",
    r"won't\b", r"don't\b", r"didn't\b", r"it's\b",
]


class StatisticalAnalyzer:

    def __init__(self):
        self.word_frequencies = self._load_word_frequencies()
        self.function_words = {
            'the','be','to','of','and','a','in','that','have','i','it','for',
            'not','on','with','he','as','you','do','at','this','but','his',
            'by','from','they','we','say','her','she','or','an','will','my',
            'one','all','would','there','their','what','so','up','out','if',
            'about','who','get','which','go','me','when','make','can','like',
            'time','no','just','him','know','take','people','into','year',
            'your','good','some','could','them','see','other','than','then',
            'now','look','only','come','its','over','think','also','back',
            'after','use','two','how','our','work','first','well','way',
            'even','new','want','because','any','these','give','day','most','us',
        }

    @staticmethod
    def _load_word_frequencies() -> Dict[str, float]:
        return {
            'the':0.061,'be':0.042,'to':0.035,'of':0.034,'and':0.031,
            'a':0.030,'in':0.028,'that':0.025,'have':0.021,'i':0.020,
            'it':0.019,'for':0.018,'not':0.017,'on':0.016,'with':0.015,
            'he':0.014,'as':0.013,'you':0.012,'do':0.011,'at':0.010,
            'this':0.009,'but':0.009,'his':0.008,'by':0.008,'from':0.008,
            'they':0.007,'we':0.007,'say':0.006,'her':0.006,'she':0.006,
            'or':0.006,'an':0.005,'will':0.005,'my':0.005,'one':0.005,
            'all':0.005,'would':0.005,'there':0.005,'their':0.005,
            'what':0.004,'so':0.004,'up':0.004,'out':0.004,'if':0.004,
            'about':0.004,'who':0.004,'get':0.004,'which':0.004,'go':0.004,
            'me':0.004,'when':0.004,'make':0.004,'can':0.004,'like':0.003,
            'time':0.003,'no':0.003,'just':0.003,'him':0.003,'know':0.003,
            'take':0.003,'people':0.003,'into':0.003,'year':0.003,
        }

    # ── main ─────────────────────────────────────────────────────────────────
    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 50:
            return self._default(50.0)

        perplexity_result    = self.calculate_perplexity(text)
        burstiness_score     = self.calculate_burstiness(text)
        repetition_score     = self.calculate_repetition_index(text)
        entropy_score        = self.calculate_entropy(text)
        zipf_score           = self.calculate_zipf_coefficient(text)
        function_word_score  = self.analyze_function_words(text)
        ai_phrase_score      = self.calculate_ai_phrase_density(text)
        human_marker_score   = self.calculate_human_marker_density(text)

        # ai_phrase_score and human_marker_score are the strongest signals
        statistical_score = (
            perplexity_result['adjusted_score'] * 0.10 +
            burstiness_score                    * 0.12 +
            repetition_score                    * 0.10 +
            entropy_score                       * 0.08 +
            zipf_score                          * 0.05 +
            function_word_score                 * 0.05 +
            ai_phrase_score                     * 0.30 +
            human_marker_score                  * 0.20
        )

        return {
            'perplexity':           perplexity_result,
            'burstiness':           burstiness_score,
            'repetition':           repetition_score,
            'entropy':              entropy_score,
            'zipf_coefficient':     zipf_score,
            'function_word_ratio':  function_word_score,
            'ai_phrase_density':    ai_phrase_score,
            'human_marker_density': human_marker_score,
            'statistical_score':    round(statistical_score, 2),
        }

    @staticmethod
    def _default(v: float) -> Dict[str, Any]:
        return {
            'perplexity':           {'raw_perplexity':v,'adjusted_perplexity':v,
                                     'final_perplexity':v,'unknown_word_ratio':0,
                                     'score':v,'adjusted_score':v},
            'burstiness':v,'repetition':v,'entropy':v,'zipf_coefficient':v,
            'function_word_ratio':v,'ai_phrase_density':v,'human_marker_density':v,
            'statistical_score':v,
        }

    @staticmethod
    def _clean_words(text: str) -> List[str]:
        return [w for w in (re.sub(r'[^\w]', '', t) for t in text.lower().split()) if w]

    # ── AI phrase density (strongest signal) ─────────────────────────────────
    def calculate_ai_phrase_density(self, text: str) -> float:
        tl   = text.lower()
        wc   = max(len(text.split()), 1)
        hits = sum(1 for p in _AI_PHRASES if p in tl)
        density = (hits / wc) * 100

        if   density == 0:   return 95.0
        elif density < 0.3:  return 88.0
        elif density < 0.7:  return 72.0
        elif density < 1.2:  return 52.0
        elif density < 2.0:  return 32.0
        elif density < 3.0:  return 18.0
        else:                return 8.0

    # ── human marker density ──────────────────────────────────────────────────
    def calculate_human_marker_density(self, text: str) -> float:
        tl  = text.lower()
        wc  = max(len(text.split()), 1)
        cnt = sum(len(re.findall(p, tl)) for p in _HUMAN_PATTERNS)
        density = (cnt / wc) * 100

        if   density > 10:  return 92.0
        elif density > 7:   return 83.0
        elif density > 5:   return 75.0
        elif density > 3:   return 62.0
        elif density > 1.5: return 48.0
        elif density > 0.5: return 35.0
        else:               return 20.0

    # ── perplexity ────────────────────────────────────────────────────────────
    def calculate_perplexity(self, text: str) -> Dict[str, float]:
        words = self._clean_words(text)
        if len(words) < 10:
            return {'raw_perplexity':50,'adjusted_perplexity':50,
                    'final_perplexity':50,'unknown_word_ratio':0,
                    'score':50,'adjusted_score':50}

        wc           = Counter(words)
        total        = len(words)
        log_prob_sum = 0.0
        unknown      = 0

        for w in words:
            prob = self.word_frequencies.get(w, max(wc[w]/total, 1e-10))
            log_prob_sum += math.log2(max(prob, 1e-10))
            if w not in self.word_frequencies:
                unknown += 1

        raw_perp     = 2 ** (-log_prob_sum / total)
        unknown_r    = (unknown / total) * 100

        bigrams      = [words[i]+'_'+words[i+1] for i in range(len(words)-1)]
        bg_counts    = Counter(bigrams)
        bg_unique_r  = len(bg_counts) / max(len(bigrams), 1)

        creativity_bonus = min(30, unknown_r * 2)
        final_perp       = raw_perp * (1 + creativity_bonus / 100)

        norm_raw   = min(100, max(0, (raw_perp  - 20) / 1.8))
        norm_final = min(100, max(0, (final_perp - 20) / 1.8))
        adjusted_score = norm_final * 0.6 + (bg_unique_r * 100) * 0.4

        return {
            'raw_perplexity':      round(raw_perp, 2),
            'adjusted_perplexity': round(final_perp, 2),
            'final_perplexity':    round(final_perp, 2),
            'unknown_word_ratio':  round(unknown_r, 2),
            'score':               round(norm_raw, 2),
            'adjusted_score':      round(min(100, adjusted_score), 2),
        }

    # ── burstiness ────────────────────────────────────────────────────────────
    def calculate_burstiness(self, text: str) -> float:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 8]
        if len(sentences) < 3:
            return 50.0

        wlens = [len(s.split()) for s in sentences]
        mean  = float(np.mean(wlens))
        std   = float(np.std(wlens))
        cv    = (std / mean) if mean > 0 else 0

        # AI: cv 0.15-0.35; humans: 0.35-0.75+
        if   cv > 0.70: return 90.0
        elif cv > 0.55: return 78.0
        elif cv > 0.40: return 65.0
        elif cv > 0.30: return 52.0
        elif cv > 0.20: return 38.0
        else:           return 22.0

    # ── repetition (completely rewritten) ─────────────────────────────────────
    def calculate_repetition_index(self, text: str) -> float:
        """
        v1 gave 92-93 to both AI and human because it measured word uniqueness
        which is always high for paragraph text. Now measures STRUCTURAL
        repetition: bigrams, trigrams, sentence openings — where AI differs.
        """
        words = self._clean_words(text)
        if len(words) < 20:
            return 50.0

        # 1. Bigram structural repetition
        bigrams = [words[i]+'_'+words[i+1] for i in range(len(words)-1)]
        if bigrams:
            bg_counts    = Counter(bigrams)
            repeated_bg  = sum(c for c in bg_counts.values() if c > 1)
            bg_rep_rate  = repeated_bg / len(bigrams)
        else:
            bg_rep_rate = 0

        # 2. Sentence opening repetition
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 8]
        if sentences:
            openers     = [' '.join(s.split()[:3]).lower() for s in sentences]
            op_counts   = Counter(openers)
            repeated_op = sum(1 for c in op_counts.values() if c > 1)
            op_rep_rate = repeated_op / len(openers)
        else:
            op_rep_rate = 0

        # 3. Trigram repetition
        if len(words) >= 3:
            trigrams  = [words[i]+'_'+words[i+1]+'_'+words[i+2] for i in range(len(words)-2)]
            tg_counts = Counter(trigrams)
            repeated_tg = sum(c-1 for c in tg_counts.values() if c > 1)
            tg_rep_rate = repeated_tg / max(len(trigrams), 1)
        else:
            tg_rep_rate = 0

        combined_rep = bg_rep_rate * 0.4 + op_rep_rate * 0.3 + tg_rep_rate * 0.3
        human_score  = max(0, min(100, (1 - combined_rep * 3) * 100))
        return round(human_score, 2)

    # ── entropy (recalibrated) ─────────────────────────────────────────────────
    def calculate_entropy(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        wc      = Counter(words)
        total   = len(words)
        entropy = -sum((c/total)*math.log2(c/total) for c in wc.values())
        # AI ~4-6 bits, human ~6-9 bits; map 3→0, 9→100
        return round(min(100, max(0, (entropy - 3) / 0.06)), 2)

    # ── Zipf ──────────────────────────────────────────────────────────────────
    def calculate_zipf_coefficient(self, text: str) -> float:
        words = self._clean_words(text)
        if len(words) < 50:
            return 50.0

        freqs    = sorted(Counter(words).values(), reverse=True)
        n_pts    = min(100, len(freqs))
        if n_pts < 2:
            return 50.0

        lr   = [math.log(r+1) for r in range(n_pts)]
        lf   = [math.log(f) for f in freqs[:n_pts]]
        n    = len(lr)
        sx   = sum(lr); sy = sum(lf)
        sxy  = sum(x*y for x,y in zip(lr,lf))
        sxx  = sum(x*x for x in lr)
        denom = n*sxx - sx*sx

        if abs(denom) < 1e-10:
            return 50.0

        slope = (n*sxy - sx*sy) / denom
        diff  = abs(slope - (-1.0))

        if   diff < 0.1: return 90.0
        elif diff < 0.2: return 75.0
        elif diff < 0.3: return 60.0
        elif diff < 0.5: return 45.0
        else:            return 30.0

    # ── function words ─────────────────────────────────────────────────────────
    def analyze_function_words(self, text: str) -> float:
        words = self._clean_words(text)
        if len(words) < 20:
            return 50.0

        ratio = (sum(1 for w in words if w in self.function_words) / len(words)) * 100

        if   45 <= ratio <= 55: return 90.0
        elif 40 <= ratio <= 60: return 80.0
        elif 35 <= ratio <= 65: return 65.0
        elif 30 <= ratio <= 70: return 50.0
        else:                   return 30.0
