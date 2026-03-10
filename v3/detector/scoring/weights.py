"""
Scoring weights v2 - Rebalanced based on which dimensions discriminate best.

Statistical and Linguistic are the strongest discriminators.
"""

class ScoringWeights:
    def __init__(self):
        self.statistical = 0.28
        self.linguistic  = 0.28
        self.content     = 0.24
        self.rhetorical  = 0.14
        self.meta        = 0.06

        self.CONFIDENCE_LEVELS = {
            'very_low':    (0,   25),
            'low':         (25,  45),
            'medium':      (45,  65),
            'high':        (65,  80),
            'very_high':   (80,  101),
        }

    def get_category_weight(self, category: str) -> float:
        return getattr(self, category, 0.0)

    def as_dict(self):
        return {
            'statistical': self.statistical,
            'linguistic':  self.linguistic,
            'content':     self.content,
            'rhetorical':  self.rhetorical,
            'meta':        self.meta,
        }
