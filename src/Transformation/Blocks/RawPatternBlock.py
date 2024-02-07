import math


class RawPatternBlock:
    NAME = "PATTERN"

    def __init__(self, ):
        pass

    def apply(self, inp):
        raise NotImplementedError


    def get_complexity(self):
        return 100


    def get_param_count(self):
        return math.nan

    def get_generalizability_index(self):
        return 0


    @classmethod
    def extract(cls, inp, blk):
        return set()

    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def __repr__(self):
        return "[Unknown pattern representation]"
