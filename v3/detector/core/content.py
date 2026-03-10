"""
Content Analyzer v2 - Rebuilt for strong AI vs Human discrimination.

Key fixes vs v1:
- authenticity: was giving 50 AI vs 95 human — good signal but formula wrong.
  AI text with no personal markers should score LOWER (5-20), not 50.
- specificity: recalibrated. AI is often vague and generic; humans cite
  specific numbers, places, names.
- Added generic_ai_content: detects "comprehensive", "holistic", "multifaceted"
  type AI padding — very strong signal.
- Added concrete_detail_score: specific facts (numbers, names, dates) = human.
"""

import re
import math
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from ..nlp_helpers import sent_tokenize, word_tokenize, count_tokens, get_entities


class ContentAnalyzer:

    def __init__(self):
        # AI overuses these abstract/generic phrases
        self.ai_generic_phrases = [
            'it is important to','it is essential to','it is crucial to',
            'it is necessary to','it is vital to','it is imperative to',
            'plays a crucial role','plays a vital role','plays an important role',
            'in today\'s world','in today\'s society','in the modern era',
            'in the digital age','in recent years','throughout history',
            'as previously mentioned','as previously stated','as noted above',
            'a wide range of','a variety of','a number of different',
            'in order to','with the aim of','for the purpose of',
            'it should be noted','it is worth noting','it is worth mentioning',
            'needless to say','it goes without saying',
            'comprehensive','holistic','multifaceted','nuanced approach',
            'paradigm shift','game changer','moving forward','going forward',
            'at the end of the day','when all is said and done',
            'in terms of','with regard to','with respect to',
            'leverage','utilize','facilitate','implement','optimize',
            'streamline','synergize','scalable','robust solution',
        ]
        self.specific_indicators = [
            r'\d+%', r'\d+ percent', r'\d+ years', r'\d+ months', r'\d+ days',
            r'\d+ people', r'\d+ million', r'\d+ billion', r'\d+ thousand',
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec',
            r'monday|tuesday|wednesday|thursday|friday|saturday|sunday',
            r'yesterday|today|tomorrow|last week|next week|last year',
        ]
        self.personal_markers = [
            'i remember','i recall','i experienced','i witnessed',
            'i encountered','i faced','i dealt with','i handled',
            'i learned','i discovered','i realized','i noticed',
            'i felt','i thought','i believed','i considered',
            'in my experience','from my experience',
            'i was','i had','i saw','i went','i did','i tried',
        ]
        self.anecdote_markers = [
            'one time','once when','there was a time','back when',
            'when i was','during my','while working on','while at',
            'it happened that','as it turned out','what happened was',
            'i remember when','back in',
        ]
        self.concrete_words = [
            'table','chair','computer','phone','book','car','house',
            'office','building','street','city','country','person',
            'manager','employee','customer','client','product','service',
            'meeting','call','email','report','document','file',
            'money','dollar','price','cost','budget','revenue','profit',
        ]
        self.abstract_words = [
            'idea','concept','theory','philosophy','principle',
            'belief','value','culture','vision','mission','strategy',
            'quality','excellence','innovation','creativity','passion',
            'success','achievement','goal','objective','target',
            'improvement','development','growth','progress','change',
        ]
        self.factual_markers = [
            'according to','based on','research shows','studies indicate',
            'data suggests','statistics show','survey found','analysis reveals',
        ]
        self.uncertainty_markers = [
            'maybe','perhaps','possibly','probably','likely',
            'i think','i believe','i suspect','i assume','i guess',
            'it seems','it appears','it might be',
        ]
        self.exaggeration_markers = [
            'always','never','everyone','no one','all','none',
            'perfect','flawless','impossible','incredible','unbelievable',
            'greatest','most important','absolutely essential',
        ]

    # ── main ──────────────────────────────────────────────────────────────────
    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 50:
            return self._default(50.0)

        scores = {
            'specificity':      self.analyze_specificity(text),
            'examples':         self.analyze_examples(text),
            'authenticity':     self.analyze_authenticity(text),
            'factual_density':  self.analyze_factual_density(text),
            'concrete_ratio':   self.analyze_concrete_abstract_ratio(text),
            'anecdotes':        self.analyze_anecdotes(text),
        }

        # authenticity is strongest signal in content dimension
        content_score = (
            scores['specificity']     * 0.15 +
            scores['examples']        * 0.15 +
            scores['authenticity']    * 0.35 +
            scores['factual_density'] * 0.15 +
            scores['concrete_ratio']  * 0.10 +
            scores['anecdotes']       * 0.10
        )

        return {
            **{k: round(v, 2) for k, v in scores.items()},
            'content_score': round(content_score, 2),
            'details': self._get_detailed_analysis(text),
        }

    @staticmethod
    def _default(v):
        keys = ['specificity','examples','authenticity','factual_density',
                'concrete_ratio','anecdotes','content_score']
        return {k: v for k in keys}

    def _get_detailed_analysis(self, text: str) -> Dict[str, Any]:
        entities_raw = get_entities(text)
        by_label: Dict[str, List[str]] = {}
        for ent_text, label in entities_raw:
            by_label.setdefault(label, [])
            if ent_text not in by_label[label]:
                by_label[label].append(ent_text)

        word_cnt, num_cnt, propn_cnt = count_tokens(text)
        return {
            'entity_count':      len(entities_raw),
            'entity_types':      {k: len(v) for k, v in by_label.items()},
            'number_count':      num_cnt,
            'proper_noun_count': propn_cnt,
            'entities_found':    by_label,
        }

    # ── specificity ───────────────────────────────────────────────────────────
    def analyze_specificity(self, text: str) -> float:
        word_cnt, num_cnt, propn_cnt = count_tokens(text)
        entities  = get_entities(text)
        ent_cnt   = len(entities)
        tl        = text.lower()

        si_cnt = sum(len(re.findall(p, tl)) for p in self.specific_indicators)
        words  = re.findall(r'\b\w+\b', tl)
        concrete_cnt = sum(1 for w in words if w in self.concrete_words)

        # AI generic phrase penalty
        gen_cnt = sum(
            len(re.findall(r'\b' + re.escape(p) + r'\b', tl))
            for p in self.ai_generic_phrases
        )

        wc  = word_cnt or 1
        nd  = (num_cnt   / wc) * 100
        pnd = (propn_cnt / wc) * 100
        end = (ent_cnt   / wc) * 100
        sid = (si_cnt    / wc) * 100
        gd  = (gen_cnt   / wc) * 100

        def band(v, thresholds, scores):
            for t, s in zip(thresholds, scores):
                if v > t: return s
            return scores[-1]

        number_score    = band(nd,  [3,2,1,0.5,0],  [90,80,70,60,50,25])
        proper_score    = band(pnd, [5,3,1,0.5,0],  [85,78,68,55,45,25])
        entity_score    = band(end, [4,2,1,0.5,0],  [85,72,62,50,40,25])
        indicator_score = band(sid, [2,1,0.5,0.2,0],[85,72,62,50,40,25])
        generic_penalty = min(40, gd * 15)

        cd = (concrete_cnt / wc) * 100
        concrete_score = 78 if cd > 2.0 else (55 if cd > 0.5 else 35)

        return round(max(0, min(100,
            number_score*0.20 + proper_score*0.20 + entity_score*0.15 +
            indicator_score*0.25 + concrete_score*0.20 - generic_penalty
        )), 2)

    # ── examples ──────────────────────────────────────────────────────────────
    def analyze_examples(self, text: str) -> float:
        tl        = text.lower()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        if len(sentences) < 3:
            return 50.0

        example_markers = [
            r'for example', r'for instance', r'such as', r'e\.g\.',
            r'like when', r'one example', r'an example',
            r'to illustrate', r'as an illustration', r'case in point',
        ]
        found = []
        for i, sent in enumerate(sentences):
            sl = sent.lower()
            for m in example_markers:
                if re.search(m, sl):
                    found.append({'text': sent, 'marker': m, 'position': i})
                    break

        if not found:
            return 38.0

        ex_scores = []
        for ex in found:
            et  = ex['text'].lower()
            s   = 50
            if any(pm in et for pm in self.personal_markers):  s += 22
            if re.search(r'\d+', et):                          s += 15
            ents = get_entities(ex['text'])
            if ents:                                            s += 15
            if sum(1 for w in et.split() if w in self.concrete_words) > 2: s += 10
            if any(am in et for am in self.anecdote_markers):  s += 15
            generic = ['something','someone','somewhere','thing','stuff']
            s -= sum(6 for g in generic if g in et)
            ex_scores.append(min(100, s))

        avg = float(np.mean(ex_scores))
        if len(found) >= 2:
            avg += min(15, len({e['marker'] for e in found}) * 5)

        return round(min(100, avg), 2)

    # ── authenticity (recalibrated — was 50 AI vs 95 human, wrong floor) ──────
    def analyze_authenticity(self, text: str) -> float:
        tl = text.lower()
        word_cnt, _, _ = count_tokens(text)
        wc = word_cnt or 1

        def density(markers):
            return (sum(len(re.findall(r'\b' + re.escape(m) + r'\b', tl))
                        for m in markers) / wc) * 100

        pd  = density(self.personal_markers)
        ad  = density(self.anecdote_markers)
        vd  = density(['struggled','challenged','difficult','hard','tough',
                       'failed','mistake','error','problem','issue',
                       'learned','grew','improved','overcame'])
        ed  = density(['felt','feeling','excited','nervous','worried',
                       'anxious','scared','happy','sad','angry',
                       'frustrated','disappointed','proud','grateful'])
        ud  = density(self.uncertainty_markers)
        xd  = density(self.exaggeration_markers)

        # AI generic content penalty (strong signal)
        gen_cnt = sum(1 for p in self.ai_generic_phrases if p in tl)
        gen_density = (gen_cnt / wc) * 100

        # Start from 20 (not 50) so AI text with no personal markers scores low
        score = 20

        for d, thresholds, bonuses in [
            (pd, [0.5, 0.2, 0], [25, 18, 8]),
            (ad, [0.5, 0.2, 0], [20, 14, 6]),
            (vd, [0.5, 0.2, 0], [18, 12, 5]),
            (ed, [0.5, 0.2, 0], [15, 10, 4]),
            (ud, [0.5, 0.2, 0], [12, 8,  3]),
        ]:
            if   d > thresholds[0]: score += bonuses[0]
            elif d > thresholds[1]: score += bonuses[1]
            elif d > 0:             score += bonuses[2]

        # Generic AI content penalty
        if   gen_density > 3: score -= 30
        elif gen_density > 2: score -= 22
        elif gen_density > 1: score -= 15
        elif gen_density > 0: score -= 8

        # Exaggeration penalty
        if   xd > 1.0: score -= 15
        elif xd > 0.5: score -= 10
        elif xd > 0.2: score -= 6

        # Specific details bonus
        fp = tl.count(' i ') + tl.count(" i'")
        score += min(10, fp * 2)
        if re.search(r'\b\d{4}\b', text):               score += 5
        if re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', text): score += 4

        return round(max(0, min(100, score)), 2)

    # ── factual density ────────────────────────────────────────────────────────
    def analyze_factual_density(self, text: str) -> float:
        tl = text.lower()
        word_cnt, num_cnt, _ = count_tokens(text)
        entities  = get_entities(text)
        ent_cnt   = len(entities)
        date_cnt  = sum(1 for _, lbl in entities if lbl in ('DATE','TIME'))
        wc        = word_cnt or 1

        fact_cnt  = sum(len(re.findall(r'\b' + re.escape(m) + r'\b', tl))
                        for m in self.factual_markers)
        stats_cnt = sum(len(re.findall(r'\b' + re.escape(ind) + r'\b', tl))
                        for ind in ['percent','statistic','average','median',
                                    'majority','minority','ratio']) + tl.count('%')

        fd  = (fact_cnt  / wc) * 100
        nd  = (num_cnt   / wc) * 100
        end = (ent_cnt   / wc) * 100
        dd  = (date_cnt  / wc) * 100
        sd  = (stats_cnt / wc) * 100

        score = 50
        score += (15 if 1<=fd<=3 else 10 if fd>3 else 5 if fd>0.5 else -8)
        score += (15 if 2<=nd<=5 else 10 if nd>5 else 5 if nd>1 else 2 if nd>0 else -8)
        score += (15 if end>3 else 10 if end>1.5 else 5 if end>0.5 else 0)
        score += (10 if dd>0.5 else 0)
        score += (10 if sd>0.2 else 0)

        op_cnt = sum(len(re.findall(r'\b' + re.escape(m) + r'\b', tl))
                     for m in ['i think','i believe','in my opinion','i feel'])
        od = (op_cnt / wc) * 100
        if 0.2 <= od <= 1.0 and fd > 0.5:
            score += 10

        return round(max(0, min(100, score)), 2)

    # ── concrete/abstract ratio ────────────────────────────────────────────────
    def analyze_concrete_abstract_ratio(self, text: str) -> float:
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 20:
            return 50.0

        conc  = sum(1 for w in words if w in self.concrete_words)
        abst  = sum(1 for w in words if w in self.abstract_words)
        ratio = (conc / abst) if abst > 0 else (conc if conc > 0 else 1)

        if   1.5 <= ratio <= 3.0: return 90.0
        elif 1.0 <= ratio < 1.5:  return 80.0
        elif 3.0 < ratio <= 5.0:  return 70.0
        elif 0.5 <= ratio < 1.0:  return 60.0
        elif ratio > 5.0:         return 50.0
        else:                     return 38.0

    # ── anecdotes ─────────────────────────────────────────────────────────────
    def analyze_anecdotes(self, text: str) -> float:
        tl        = text.lower()
        sentences = sent_tokenize(text)
        if not sentences:
            return 25.0

        an_cnt = sum(len(re.findall(r'\b' + re.escape(m) + r'\b', tl))
                     for m in self.anecdote_markers)

        story_elem = {
            'intro': ['when','while','during','back in'],
            'dev':   ['then','next','after','later','subsequently'],
            'conc':  ['finally','ultimately','in the end','eventually'],
        }
        elem_counts = {
            k: sum(len(re.findall(r'\b' + re.escape(m) + r'\b', tl)) for m in mlist)
            for k, mlist in story_elem.items()
        }

        density = (an_cnt / len(sentences)) * 10

        if   density > 2.0: score = 88
        elif density > 1.0: score = 78
        elif density > 0.5: score = 68
        elif density > 0.2: score = 55
        elif density > 0:   score = 42
        else:               score = 22

        if all(c > 0 for c in elem_counts.values()): score += 15
        elif sum(elem_counts.values()) > 3:           score += 10
        elif sum(elem_counts.values()) > 1:           score += 5
        if an_cnt >= 2:                               score += 8

        return round(min(100, score), 2)
