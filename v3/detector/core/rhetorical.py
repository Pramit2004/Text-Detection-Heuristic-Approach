"""
Rhetorical Analyzer v2 - Rebuilt for strong AI vs Human discrimination.

Key fixes vs v1:
- personal_voice: was giving 56 AI vs 58 human — nearly useless. Complete rewrite.
  Now properly detects first-person authentic voice vs third-person formal AI voice.
- emotional_language: was giving 18 to both (floor effect). Now detects raw,
  authentic emotional vocabulary vs AI's sanitised "positive/negative" framing.
- storytelling: improved to detect narrative flow signals more reliably.
- Added imperfection_signals: typos, self-corrections, tangents = human signals.
- Added formality_score: academic/formal language = AI signal.
"""

import re
import math
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from ..nlp_helpers import sent_tokenize, word_tokenize


class RhetoricalAnalyzer:

    def __init__(self):
        self.personal_pronouns = {
            'first_person_singular': ['i','me','my','mine','myself'],
            'first_person_plural':   ['we','us','our','ours','ourselves'],
            'second_person':         ['you','your','yours','yourself'],
            'third_person':          ['he','him','his','she','her','hers',
                                      'it','its','they','them','their'],
        }
        self.opinion_markers = [
            'i believe','in my opinion','i think','i feel','i consider',
            'from my perspective','it seems to me','i would argue',
            'i am convinced','i doubt','i suspect','to me',
            'personally','in my view','i reckon','i find',
        ]
        self.claim_markers     = ['because','since','as','due to','therefore',
                                  'thus','hence','consequently','so','accordingly']
        self.evidence_markers  = ['for example','for instance','such as','specifically',
                                  'in particular','to illustrate','as evidence',
                                  'according to','based on']
        self.conclusion_markers= ['in conclusion','to conclude','in summary',
                                  'to summarize','overall','ultimately','finally',
                                  'in the end','all things considered']

        # Raw human emotions — not the sanitised list AI uses
        self.raw_emotion_words = [
            'awful','terrible','horrible','disgusting','shocking','outraged',
            'furious','pissed','scared','freaked','panicked','devastated',
            'ecstatic','thrilled','pumped','stoked','gutted','crushed',
            'awkward','embarrassing','cringe','weird','bizarre','insane',
            'laughed','cried','screamed','yelled','whispered','muttered',
            'rushed','scrambled','stumbled','hesitated','froze',
            'loved','hated','missed','regretted','wished','hoped',
            'annoyed','irritated','exhausted','overwhelmed','confused',
        ]
        # Formal/academic language = AI signal
        self.formal_words = [
            'furthermore','moreover','consequently','subsequently',
            'aforementioned','utilizing','facilitate','demonstrate',
            'illustrate','pertaining','regarding','concerning',
            'endeavor','commence','terminate','implement','establish',
            'significant','substantial','fundamental','comprehensive',
            'methodology','framework','paradigm','infrastructure',
            'stakeholders','leverage','synergy','optimize',
        ]
        self.storytelling_markers = [
            'when i','i remember','i recall','back then','one time',
            'there was a time','i experienced','i went through',
            'i encountered','it happened','after that','i was',
            'last week','last year','yesterday','this morning',
        ]
        self.imperfection_markers = [
            r'\.\.\.',       # ellipsis = trailing thought
            r'\bwait\b',     # self-correction
            r'\bactually\b', # self-correction
            r'\banyway\b',   # tangent return
            r'\bsorry\b',    # social correction
            r'\boh\b',       # realisation
            r'\buh\b',       # filler
            r'\bum\b',       # filler
            r'\bhm\b',
            r'\bhmm\b',
        ]

    # ── main ──────────────────────────────────────────────────────────────────
    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 50:
            return self._default(50.0)

        scores = {
            'personal_voice':       self.analyze_personal_voice(text),
            'argument_structure':   self.analyze_argument_structure(text),
            'tone_consistency':     self.analyze_tone_consistency(text),
            'emotional_language':   self.analyze_emotional_language(text),
            'storytelling':         self.analyze_storytelling(text),
            'rhetorical_questions': self.analyze_rhetorical_questions(text),
        }

        rhetorical_score = (
            scores['personal_voice']       * 0.30 +
            scores['argument_structure']   * 0.15 +
            scores['tone_consistency']     * 0.15 +
            scores['emotional_language']   * 0.20 +
            scores['storytelling']         * 0.15 +
            scores['rhetorical_questions'] * 0.05
        )

        return {
            **{k: round(v, 2) for k, v in scores.items()},
            'rhetorical_score': round(rhetorical_score, 2),
            'details': self._get_detailed_analysis(text),
        }

    @staticmethod
    def _default(v):
        keys = ['personal_voice','argument_structure','tone_consistency',
                'emotional_language','storytelling','rhetorical_questions',
                'rhetorical_score']
        return {k: v for k in keys}

    def _get_detailed_analysis(self, text: str) -> Dict[str, Any]:
        tl = text.lower()
        pronoun_counts = {}
        for cat, prons in self.personal_pronouns.items():
            pronoun_counts[cat] = sum(
                len(re.findall(r'\b' + p + r'\b', tl)) for p in prons)
        emo = sum(len(re.findall(r'\b' + w + r'\b', tl)) for w in self.raw_emotion_words)
        return {
            'sentence_count':      len(sent_tokenize(text)),
            'personal_pronouns':   pronoun_counts,
            'question_count':      text.count('?'),
            'emotional_word_count':emo,
            'has_storytelling':    any(m in tl for m in self.storytelling_markers),
        }

    # ── personal voice (completely rewritten) ─────────────────────────────────
    def analyze_personal_voice(self, text: str) -> float:
        """
        v1 gave 56 AI vs 58 human — completely useless. Rewritten to detect
        the specific pattern: AI uses 3rd person / impersonal framing;
        humans use 1st person and express opinions directly.
        """
        tl    = text.lower()
        words = [w for w in word_tokenize(tl) if w.isalnum()]
        if len(words) < 20:
            return 50.0

        # Count 1st person singular — the most personal signal
        fp_count = sum(len(re.findall(r'\b' + p + r'\b', tl))
                       for p in self.personal_pronouns['first_person_singular'])
        fp_density = (fp_count / len(words)) * 100

        # Opinion markers (strong human signal)
        op_count = sum(len(re.findall(r'\b' + re.escape(m) + r'\b', tl))
                       for m in self.opinion_markers)
        op_density = (op_count / len(words)) * 100

        # Formal vocabulary (AI signal — penalise)
        formal_count = sum(len(re.findall(r'\b' + re.escape(w) + r'\b', tl))
                           for w in self.formal_words)
        formal_density = (formal_count / len(words)) * 100

        # Imperfection markers (human signal)
        imperf_count = sum(len(re.findall(p, tl)) for p in self.imperfection_markers)
        imperf_density = (imperf_count / len(words)) * 100

        # 1st person score
        if   fp_density > 5:   fp_score = 90
        elif fp_density > 3:   fp_score = 80
        elif fp_density > 1.5: fp_score = 65
        elif fp_density > 0.5: fp_score = 48
        else:                  fp_score = 20

        # Opinion score
        if   op_density > 1.0: op_score = 88
        elif op_density > 0.5: op_score = 75
        elif op_density > 0.2: op_score = 58
        elif op_density > 0:   op_score = 42
        else:                  op_score = 20

        # Formal penalty
        if   formal_density > 3: formal_penalty = 35
        elif formal_density > 2: formal_penalty = 25
        elif formal_density > 1: formal_penalty = 15
        else:                    formal_penalty = 0

        # Imperfection bonus
        imperf_bonus = min(15, imperf_count * 5)

        score = fp_score*0.50 + op_score*0.40 + imperf_bonus - formal_penalty*0.10
        return round(max(0, min(100, score)), 2)

    # ── argument structure ─────────────────────────────────────────────────────
    def analyze_argument_structure(self, text: str) -> float:
        sentences = sent_tokenize(text)
        if len(sentences) < 5:
            return 50.0
        tl = text.lower()

        claim_cnt    = sum(len(re.findall(r'\b'+re.escape(m)+r'\b', tl)) for m in self.claim_markers)
        evidence_cnt = sum(len(re.findall(r'\b'+re.escape(m)+r'\b', tl)) for m in self.evidence_markers)
        conc_cnt     = sum(len(re.findall(r'\b'+re.escape(m)+r'\b', tl)) for m in self.conclusion_markers)

        ns  = len(sentences)
        cr  = claim_cnt    / ns
        er  = evidence_cnt / ns

        claim_score    = (85 if 0.2<=cr<=0.5 else 70 if 0.1<=cr<0.2 else 60 if 0.5<cr<=0.8 else 40)
        evidence_score = (85 if er>=0.15 else 75 if er>=0.10 else 60 if er>=0.05 else 30)

        if claim_cnt > 0:
            cer = evidence_cnt / claim_cnt
            balance_score = (90 if 0.3<=cer<=0.8 else 75 if 0.2<=cer<0.3 else 70 if 0.8<cer<=1.2 else 50)
        else:
            balance_score = 40

        if conc_cnt > 0 and ns > 3:
            positions   = [i for i, s in enumerate(sentences)
                           if any(m in s.lower() for m in self.conclusion_markers)]
            avg_pos     = float(np.mean(positions)) / ns if positions else 0
            position_score = 85 if avg_pos > 0.7 else 70 if avg_pos > 0.5 else 50
        else:
            position_score = 60

        return round(claim_score*0.25 + evidence_score*0.30 + balance_score*0.30 + position_score*0.15, 2)

    # ── tone consistency ───────────────────────────────────────────────────────
    def analyze_tone_consistency(self, text: str) -> float:
        sentences = sent_tokenize(text)
        if len(sentences) < 5:
            return 50.0

        openings = []
        for sent in sentences:
            words = sent.split()
            if not words:
                continue
            op = (' '.join(words[:2]) if len(words) >= 2 else words[0]).lower().rstrip('.,!?;:')
            openings.append(op)

        if not openings:
            return 50.0

        variety_score = (len(set(openings)) / len(openings)) * 100
        rep_counts    = Counter(openings)
        rep_penalty   = (sum(1 for c in rep_counts.values() if c > 1) / len(openings)) * 50

        lengths = [len(s.split()) for s in sentences if s.split()]
        if len(lengths) > 1:
            changes = [abs(lengths[i]-lengths[i-1]) for i in range(1, len(lengths))]
            avg_c   = float(np.mean(changes)) if changes else 0
            std_c   = float(np.std(changes))  if changes else 0
            if   std_c < 2 and avg_c < 3: change_score = 25
            elif std_c < 3 and avg_c < 4: change_score = 45
            elif std_c < 4 and avg_c < 5: change_score = 68
            elif std_c < 6:               change_score = 82
            else:                         change_score = 72
        else:
            change_score = 50

        return round(min(100, variety_score*0.30 + (100-rep_penalty)*0.25 + change_score*0.45), 2)

    # ── emotional language (completely rewritten) ──────────────────────────────
    def analyze_emotional_language(self, text: str) -> float:
        """
        v1 gave 18 to BOTH AI and human — complete floor effect.
        Rewritten to detect raw/authentic emotional language (human)
        vs sanitised/formal emotional language (AI).
        """
        tl    = text.lower()
        words = [w for w in word_tokenize(tl) if w.isalnum()]
        if len(words) < 20:
            return 50.0

        # Raw human emotions
        raw_count  = sum(len(re.findall(r'\b' + w + r'\b', tl))
                         for w in self.raw_emotion_words)
        raw_density = (raw_count / len(words)) * 100

        # Exclamation marks = emotional expression
        exclaim_density = (text.count('!') / len(words)) * 100

        # Capitalisation for emphasis (ALL CAPS words)
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        caps_density = (caps_words / len(words)) * 100

        # Informal intensifiers
        intensifiers = ['really','very','so','totally','absolutely',
                        'completely','definitely','seriously','literally',
                        'actually','honestly','genuinely']
        int_count   = sum(len(re.findall(r'\b' + w + r'\b', tl)) for w in intensifiers)
        int_density = (int_count / len(words)) * 100

        # Score each component
        if   raw_density > 2.0:  raw_score = 88
        elif raw_density > 1.0:  raw_score = 75
        elif raw_density > 0.5:  raw_score = 60
        elif raw_density > 0.2:  raw_score = 45
        else:                    raw_score = 22

        exclaim_score = min(85, exclaim_density * 400)
        caps_score    = min(75, caps_density * 500)
        if   int_density > 3:   int_score = 80
        elif int_density > 1.5: int_score = 68
        elif int_density > 0.5: int_score = 52
        else:                   int_score = 30

        return round(raw_score*0.45 + exclaim_score*0.15 + caps_score*0.10 + int_score*0.30, 2)

    # ── storytelling ──────────────────────────────────────────────────────────
    def analyze_storytelling(self, text: str) -> float:
        tl        = text.lower()
        sentences = sent_tokenize(text)
        if len(sentences) < 5:
            return 50.0

        st_count = sum(len(re.findall(r'\b' + re.escape(m) + r'\b', tl))
                       for m in self.storytelling_markers)
        time_markers = ['first','then','next','after','later','finally',
                        'before','during','while','when','afterwards',
                        'suddenly','immediately','eventually']
        t_count  = sum(len(re.findall(r'\b' + m + r'\b', tl)) for m in time_markers)
        exp_markers = ['i went','i saw','i heard','i felt','i noticed',
                       'i found','i met','i tried','i had','i was']
        e_count  = sum(len(re.findall(r'\b' + re.escape(m) + r'\b', tl)) for m in exp_markers)

        # Specific dates/times = narrative anchoring (human)
        has_specific_time = bool(re.search(
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
            r'january|february|march|april|may|june|july|august|september|'
            r'october|november|december|\d{4}|last week|yesterday|today|'
            r'this morning|last night)\b', tl))

        sd = (st_count / len(sentences)) * 10

        if   sd > 2.0: score = 88
        elif sd > 1.0: score = 78
        elif sd > 0.5: score = 65
        elif sd > 0.2: score = 52
        elif sd > 0:   score = 42
        else:          score = 25

        if e_count > 0:        score += min(15, e_count * 5)
        if t_count > 3:        score += 8
        if has_specific_time:  score += 10

        return round(min(100, score), 2)

    # ── rhetorical questions ───────────────────────────────────────────────────
    def analyze_rhetorical_questions(self, text: str) -> float:
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return 50.0

        tl = text.lower()
        rhet_ind = ['who would','what if','why would','how could',
                    "isn't it","aren't we","wouldn't you",
                    "don't you","can't we","could it be",
                    'right?','you know?']
        rhet_cnt = sum(len(re.findall(r'\b' + re.escape(r) + r'\b', tl)) for r in rhet_ind)

        q_density = (text.count('?') / len(sentences)) * 10

        if   q_density > 2.0: score = 82
        elif q_density > 1.0: score = 75
        elif q_density > 0.5: score = 65
        elif q_density > 0.2: score = 55
        elif q_density > 0:   score = 45
        else:                 score = 30

        if rhet_cnt > 0: score += 10
        if rhet_cnt > 1: score += 5

        return round(min(100, score), 2)
