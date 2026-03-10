"""
Scoring Calculator — fixed version.

Bugs fixed:
 • meta_results['meta_score'] was being fetched correctly, but
   meta_results['ai_tool_mentions'] is a *dict* (not a float); report
   generation accessed it as a float in several places → added safe .get().
 • category_scores used `meta_results['meta_score']` but the key coming back
   from MetaAnalyzer was already 'meta_score' — this was correct, but the
   sub-key 'ai_tool_mentions' is a nested dict; all report references fixed.
 • _calculate_confidence: np.var on empty list guarded.
 • _generate_report: bar graph string concat with non-int bar_length fixed
   (was float from score/5).
"""

import datetime
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.statistical import StatisticalAnalyzer
from ..core.linguistic  import LinguisticAnalyzer
from ..core.rhetorical  import RhetoricalAnalyzer
from ..core.content     import ContentAnalyzer
from ..core.meta        import MetaAnalyzer
from .weights           import ScoringWeights


class GrandpaDetector:

    def __init__(self):
        self.statistical = StatisticalAnalyzer()
        self.linguistic  = LinguisticAnalyzer()
        self.rhetorical  = RhetoricalAnalyzer()
        self.content     = ContentAnalyzer()
        self.meta        = MetaAnalyzer()
        self.weights     = ScoringWeights()

    # ── public API ────────────────────────────────────────────────────────────
    def analyze(self, text: str, filename: Optional[str] = None) -> Dict[str, Any]:
        if not text or len(text.strip()) < 50:
            return self._get_insufficient_text_response()

        stat_r  = self.statistical.analyze(text)
        ling_r  = self.linguistic.analyze(text)
        rhet_r  = self.rhetorical.analyze(text)
        cont_r  = self.content.analyze(text)
        meta_r  = self.meta.analyze(text, filename)

        category_scores = {
            'statistical': stat_r['statistical_score'],
            'linguistic':  ling_r['linguistic_score'],
            'rhetorical':  rhet_r['rhetorical_score'],
            'content':     cont_r['content_score'],
            'meta':        meta_r['meta_score'],
        }

        total_score  = self._calculate_weighted_score(category_scores)
        confidence   = self._calculate_confidence(category_scores)
        classification = self._classify(total_score)

        detailed = {
            'statistical': stat_r,
            'linguistic':  ling_r,
            'rhetorical':  rhet_r,
            'content':     cont_r,
            'meta':        meta_r,
        }

        return {
            'total_score':      round(total_score, 2),
            'confidence':       round(confidence, 2),
            'confidence_level': self._get_confidence_level(confidence),
            'category_scores':  {k: round(v, 2) for k, v in category_scores.items()},
            'classification':   classification,
            'detailed_metrics': detailed,
            'report':           self._generate_report(text, category_scores, total_score,
                                                       classification, detailed),
            'suggestions':      self._generate_suggestions(category_scores, detailed),
            'metadata': {
                'analyzed_at': datetime.datetime.now().isoformat(),
                'text_length': len(text),
                'word_count':  len(text.split()),
                'filename':    filename,
            },
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze(t) for t in texts]

    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        r1 = self.analyze(text1)
        r2 = self.analyze(text2)
        return {
            'text1_score':        r1['total_score'],
            'text2_score':        r2['total_score'],
            'difference':         abs(r1['total_score'] - r2['total_score']),
            'text1_classification':r1['classification']['label'],
            'text2_classification':r2['classification']['label'],
            'more_human_like':    'text1' if r1['total_score'] > r2['total_score'] else 'text2',
            'category_comparison': {
                cat: {
                    'text1':     r1['category_scores'][cat],
                    'text2':     r2['category_scores'][cat],
                    'difference':abs(r1['category_scores'][cat] - r2['category_scores'][cat]),
                }
                for cat in r1['category_scores']
            },
        }

    # ── scoring helpers ───────────────────────────────────────────────────────
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        return sum(v * self.weights.get_category_weight(k) for k, v in scores.items())

    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        vals = list(scores.values())
        if not vals:
            return 0.0

        variance    = float(np.var(vals))           # ← FIX: guarded empty list above
        score_range = max(vals) - min(vals)
        extremes    = sum(1 for s in vals if s < 20 or s > 80)

        base = 80
        base -= 30 if variance > 500 else (20 if variance > 300 else 10 if variance > 150 else 0)
        base -= 20 if score_range > 50 else (10 if score_range > 30 else 0)
        base -= extremes * 5
        return max(0, min(100, base))

    def _get_confidence_level(self, confidence: float) -> str:
        for level, (lo, hi) in self.weights.CONFIDENCE_LEVELS.items():
            if lo <= confidence < hi:
                return level.replace('_', ' ').title()
        return 'Medium'

    def _classify(self, score: float) -> Dict[str, Any]:
        if score < 30:
            return dict(label='AI-Generated', confidence='High', color='red',
                        emoji='🤖',
                        description='This text shows strong patterns of AI generation.',
                        details='Uniform structure, repetitive patterns, lack of personal voice.')
        if score < 45:
            return dict(label='Likely AI-Generated', confidence='Medium-High', color='orange',
                        emoji='⚙️',
                        description='Moderate to strong AI patterns detected.',
                        details='Several AI-like characteristics but some human elements present.')
        if score < 65:
            return dict(label='Mixed / Uncertain', confidence='Low', color='#f59e0b',
                        emoji='🔄',
                        description='Could be human writing with AI assistance, or vice versa.',
                        details='Mix of human and AI characteristics detected.')
        if score < 80:
            return dict(label='Likely Human-Written', confidence='Medium-High', color='#10b981',
                        emoji='👤',
                        description='Strong human writing patterns detected.',
                        details='Varied structure, personal voice, and natural flow observed.')
        return dict(label='Human-Written', confidence='High', color='green',
                    emoji='👨‍💼',
                    description='Clear human writing characteristics.',
                    details='Personal experiences, varied style, emotional depth, and natural inconsistencies.')

    # ── report generation ─────────────────────────────────────────────────────
    def _generate_report(self, text, category_scores, total_score,
                          classification, detailed) -> str:
        R = []
        R.append("=" * 60)
        R.append("👴 GRANDPA TEXT DETECTION REPORT")
        R.append("=" * 60)
        R.append("")
        R.append(f"📊 OVERALL HUMAN SCORE: {total_score:.1f}/100")
        R.append(f"📌 CLASSIFICATION: {classification['emoji']} {classification['label']}")
        R.append(f"📝 {classification['description']}")
        R.append("")

        conf = self._calculate_confidence(category_scores)
        R.append(f"🎯 CONFIDENCE: {conf:.1f}% ({self._get_confidence_level(conf)})")
        R.append("")
        R.append("📈 CATEGORY BREAKDOWN:")
        R.append("")

        for cat, score in category_scores.items():
            bar_len  = int(score / 5)           # ← FIX: explicit int cast
            bar      = "█" * bar_len + "░" * (20 - bar_len)
            R.append(f"  {cat.replace('_',' ').title():12}: {bar} {score:.1f}")

        R.append("")
        R.append("-" * 60)
        R.append("")
        R.append("🔍 DETAILED ANALYSIS:")
        R.append("")

        # Statistical
        stat = detailed['statistical']
        perp = stat.get('perplexity', {})
        perp_score = perp.get('adjusted_score', 50) if isinstance(perp, dict) else 50
        R.append("  📊 Statistical Analysis:")
        R.append(f"    • Perplexity:  {perp_score:.1f}/100 – "
                 f"{'High variation' if perp_score>70 else 'Low variation' if perp_score<40 else 'Moderate'}")
        R.append(f"    • Burstiness:  {stat.get('burstiness',50):.1f}/100")
        R.append(f"    • Repetition:  {stat.get('repetition',50):.1f}/100")
        R.append("")

        # Linguistic
        ling = detailed['linguistic']
        R.append("  🗣️ Linguistic Analysis:")
        R.append(f"    • Vocabulary:        {ling.get('vocabulary_diversity',50):.1f}/100")
        R.append(f"    • Sentence Structure: {ling.get('sentence_structure',50):.1f}/100")
        R.append(f"    • Hedge Words:       {ling.get('hedge_words',50):.1f}/100")
        R.append("")

        # Rhetorical
        rhet = detailed['rhetorical']
        R.append("  🎭 Rhetorical Analysis:")
        R.append(f"    • Personal Voice:    {rhet.get('personal_voice',50):.1f}/100")
        R.append(f"    • Emotional Language:{rhet.get('emotional_language',50):.1f}/100")
        R.append(f"    • Storytelling:      {rhet.get('storytelling',50):.1f}/100")
        R.append("")

        # Content
        cont = detailed['content']
        R.append("  📝 Content Analysis:")
        R.append(f"    • Specificity:  {cont.get('specificity',50):.1f}/100")
        R.append(f"    • Authenticity: {cont.get('authenticity',50):.1f}/100")
        R.append(f"    • Examples:     {cont.get('examples',50):.1f}/100")
        R.append("")

        # Meta — ai_tool_mentions is a dict  ← FIX
        meta     = detailed['meta']
        ai_info  = meta.get('ai_tool_mentions', {})
        ai_cnt   = ai_info.get('count', 0) if isinstance(ai_info, dict) else 0
        tmpl_info= meta.get('template_indicators', {})
        tmpl_cnt = tmpl_info.get('count', 0) if isinstance(tmpl_info, dict) else 0
        R.append("  🔖 Meta Analysis:")
        R.append(f"    • AI Tool Mentions:   {ai_cnt} – {'Suspicious' if ai_cnt>0 else 'Clean'}")
        R.append(f"    • Template Patterns:  {tmpl_cnt} found")
        R.append(f"    • Formatting:         {meta.get('formatting_patterns',50):.1f}/100")
        R.append("")
        R.append("-" * 60)
        R.append("")
        R.append("⚠️ KEY INDICATORS:")
        R.append("")

        indicators = self._collect_indicators(detailed)
        if indicators['ai_indicators']:
            R.append("  AI Indicators:")
            for ind in indicators['ai_indicators'][:5]:
                R.append(f"    • {ind}")
        if indicators['human_indicators']:
            R.append("")
            R.append("  Human Indicators:")
            for ind in indicators['human_indicators'][:5]:
                R.append(f"    • {ind}")

        R.append("")
        R.append("=" * 60)
        return "\n".join(R)

    def _collect_indicators(self, detailed: Dict[str, Any]) -> Dict[str, List[str]]:
        indicators: Dict[str, List[str]] = {'ai_indicators': [], 'human_indicators': []}

        def add(condition_ai, condition_human, ai_msg, human_msg):
            if condition_ai:   indicators['ai_indicators'].append(ai_msg)
            elif condition_human: indicators['human_indicators'].append(human_msg)

        stat = detailed.get('statistical', {})
        perp = stat.get('perplexity', {})
        ps   = perp.get('adjusted_score', 50) if isinstance(perp, dict) else 50
        add(ps < 40, ps > 70, "Low perplexity (too predictable)", "High perplexity (good variation)")
        add(stat.get('burstiness',50) < 40, stat.get('burstiness',50) > 70,
            "Low burstiness (uniform sentences)", "High burstiness (natural rhythm)")
        add(stat.get('repetition',50) < 40, stat.get('repetition',50) > 70,
            "High repetition (repetitive patterns)", "Low repetition (good variety)")

        ling = detailed.get('linguistic', {})
        add(ling.get('vocabulary_diversity',50) < 40,
            ling.get('vocabulary_diversity',50) > 70,
            "Limited vocabulary", "Rich vocabulary")
        add(ling.get('hedge_words',50) < 30, ling.get('hedge_words',50) > 60,
            "No uncertainty markers (overly confident)", "Natural uncertainty (uses hedge words)")

        rhet = detailed.get('rhetorical', {})
        add(rhet.get('personal_voice',50) < 40, rhet.get('personal_voice',50) > 70,
            "Lacks personal voice", "Strong personal voice")
        add(rhet.get('emotional_language',50) < 40,
            rhet.get('emotional_language',50) > 70,
            "Emotionally flat", "Emotionally rich")

        cont = detailed.get('content', {})
        add(cont.get('specificity',50) < 40, cont.get('specificity',50) > 70,
            "Generic content (lacks specifics)", "Specific details and examples")
        add(cont.get('authenticity',50) < 40, cont.get('authenticity',50) > 70,
            "Formulaic/inauthentic", "Authentic personal voice")

        # FIX: ai_tool_mentions is a dict
        meta    = detailed.get('meta', {})
        ai_info = meta.get('ai_tool_mentions', {})
        ai_cnt  = ai_info.get('count', 0) if isinstance(ai_info, dict) else 0
        if ai_cnt > 0:
            indicators['ai_indicators'].append(f"Mentions AI tools ({ai_cnt} times)")

        tmpl = meta.get('template_indicators', {})
        tc   = tmpl.get('count', 0) if isinstance(tmpl, dict) else 0
        if tc > 0:
            indicators['ai_indicators'].append(f"Contains template patterns ({tc} found)")

        return indicators

    def _generate_suggestions(self, category_scores: Dict[str, float],
                               detailed: Dict[str, Any]) -> List[Dict[str, Any]]:
        sugg = []

        def add(category, priority, issue, suggestion):
            sugg.append({'category': category, 'priority': priority,
                         'issue': issue, 'suggestion': suggestion})

        # Statistical
        if category_scores.get('statistical', 100) < 60:
            stat = detailed.get('statistical', {})
            perp = stat.get('perplexity', {})
            ps   = perp.get('adjusted_score', 50) if isinstance(perp, dict) else 50
            if ps < 40:
                add('statistical','high','Text is too predictable',
                    'Vary sentence structure and word choice. Add unexpected elements.')
            if stat.get('burstiness', 50) < 40:
                add('statistical','medium','Sentences are too uniform in length',
                    'Mix short and long sentences intentionally.')
            if stat.get('repetition', 50) < 40:
                add('statistical','high','Repetitive language patterns',
                    'Use synonyms and vary phrasing. Avoid repeating the same structures.')

        # Linguistic
        if category_scores.get('linguistic', 100) < 60:
            ling = detailed.get('linguistic', {})
            if ling.get('vocabulary_diversity', 50) < 40:
                add('linguistic','high','Limited vocabulary',
                    'Use more descriptive words; avoid repeating common terms.')
            if ling.get('hedge_words', 50) < 30:
                add('linguistic','medium','Too certain/absolute',
                    'Add uncertainty markers: "perhaps", "maybe", "I think".')
            if ling.get('transition_usage', 50) < 40:
                add('linguistic','medium','Overuse of transition words',
                    'Reduce "however", "therefore", "furthermore". Let ideas flow naturally.')

        # Rhetorical
        if category_scores.get('rhetorical', 100) < 60:
            rhet = detailed.get('rhetorical', {})
            if rhet.get('personal_voice', 50) < 40:
                add('rhetorical','high','Lacks personal voice',
                    'Use personal pronouns (I, me, my) and share your perspective.')
            if rhet.get('emotional_language', 50) < 40:
                add('rhetorical','medium','Emotionally flat',
                    'Include emotional language where appropriate.')
            if rhet.get('storytelling', 50) < 30:
                add('rhetorical','medium','No storytelling elements',
                    'Include brief anecdotes or personal stories.')

        # Content
        if category_scores.get('content', 100) < 60:
            cont = detailed.get('content', {})
            if cont.get('specificity', 50) < 40:
                add('content','high','Generic content',
                    'Add specific details: numbers, dates, names, places.')
            if cont.get('authenticity', 50) < 40:
                add('content','high','Formulaic or inauthentic',
                    'Share genuine experiences including challenges and lessons learned.')
            if cont.get('examples', 50) < 40:
                add('content','medium','Missing or generic examples',
                    'Include specific personal examples instead of generic illustrations.')

        # Meta
        if category_scores.get('meta', 100) < 60:
            meta    = detailed.get('meta', {})
            ai_info = meta.get('ai_tool_mentions', {})
            ai_cnt  = ai_info.get('count', 0) if isinstance(ai_info, dict) else 0
            if ai_cnt > 0:
                add('meta','high','Mentions AI tools',
                    'Remove mentions of AI tools or generators.')
            tmpl = meta.get('template_indicators', {})
            tc   = tmpl.get('count', 0) if isinstance(tmpl, dict) else 0
            if tc > 0:
                add('meta','high','Contains template patterns',
                    'Remove placeholder text like [Your Name]. Personalise all sections.')

        sugg.sort(key=lambda x: {'high':0,'medium':1,'low':2}.get(x['priority'],3))
        return sugg

    def _get_insufficient_text_response(self) -> Dict[str, Any]:
        return {
            'total_score': 50.0, 'confidence': 0, 'confidence_level': 'Very Low',
            'category_scores': {k: 50.0 for k in
                                 ['statistical','linguistic','rhetorical','content','meta']},
            'classification': {
                'label': 'Insufficient Text', 'confidence': 'N/A', 'color': 'gray',
                'emoji': '⚠️',
                'description': 'Text is too short for reliable analysis (minimum 50 characters)',
                'details': 'Please provide at least 50 characters for accurate detection.',
            },
            'detailed_metrics': {},
            'report': '⚠️ INSUFFICIENT TEXT: Please provide at least 50 characters.',
            'suggestions': [{'category':'general','priority':'high',
                             'issue':'Text too short',
                             'suggestion':'Provide at least 50 characters for meaningful analysis.'}],
            'metadata': {
                'analyzed_at': datetime.datetime.now().isoformat(),
                'text_length': 0,'word_count': 0,'filename': None,
            },
        }
