import difflib

import numpy as np
from rapidfuzz import fuzz


class StringMatcher:
    def __init__(self) -> None:
        self.accum_exact = []
        self.accum_fuzzy = []

    def exact(self, str1, str2) -> None:
        seq = difflib.SequenceMatcher(None, str1, str2)
        score = seq.ratio() * 100
        self.accum_exact.append(round(score, 2))

    def fuzzy(self, str1, str2) -> None:
        self.accum_fuzzy.append(fuzz.ratio(str1, str2))

    def get_exact(self):
        return np.mean(self.accum_exact)

    def get_fuzzy(self):
        return np.mean(self.accum_fuzzy)
