from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class SubstrG (RawPatternBlock):
    NAME = "SUBSTRG"

    def __init__(self, start=None, end=None):
        # To init hash:
        self._hash = None
        self._start = None
        self._end = None

        self.start = start
        self.end = end

    def update_hash(self):
        self._hash = hash((self.NAME, self.start, self.end))

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        self._start = start
        self.update_hash()

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, end):
        self._end = end
        self.update_hash()

    def apply(self, inp):
        n = len(inp)

        if self.start.startswith('e-'):
            tmp = self.start.split('-')
            assert len(tmp) == 2 and tmp[0] == 'e'
            real_start = n - int(tmp[1])

        elif self.start.startswith('s+'):
            tmp = self.start.split('+')
            assert len(tmp) == 2 and tmp[0] == 's'
            real_start = int(tmp[1])
        else:
            print(f"Wrong params for start index of Substr: {self.start}")
            raise IndexError("Wrong start")


        if self.end.startswith('e-'):
            tmp = self.end.split('-')
            assert len(tmp) == 2 and tmp[0] == 'e'
            real_end = n - int(tmp[1])
        elif self.end.startswith('s+'):
            tmp = self.end.split('+')
            assert len(tmp) == 2 and tmp[0] == 's'
            real_end = int(tmp[1])
        else:
            print(f"Wrong params for end index of Substr: {self.end}")
            raise IndexError("Wrong end")




        if real_end > len(inp):
            raise IndexError("end > len(input)")
        return inp[real_start:real_end]


    def get_complexity(self):
        return 2


    def get_param_count(self):
        return 2

    def get_generalizability_index(self):
        lst = [self.start, self.end]
        return sum(1 if (l == 'e-0' or l == 's+0') else 0 for l in lst)

    @classmethod
    def extract(cls, inp, blk):
        s = set()
        st1 = f"e-{len(inp)-blk.start}"
        st2 = f"s+{blk.start}"
        en1 = f"e-{len(inp)-blk.end}"
        en2 = f"s+{blk.end}"

        s.add(SubstrG(st1, en1))
        s.add(SubstrG(st1, en2))
        s.add(SubstrG(st2, en1))
        s.add(SubstrG(st2, en2))

        return s

    @classmethod
    def get_param_space(cls, inp_lst):
        raise NotImplementedError

    @classmethod
    def get_random(cls, inp_charset, input_max_len):
        raise NotImplementedError

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return self._hash
        # return hash((self.start, self.end,))

    def __repr__(self):
        return f"[SubstrG:({self.start},{self.end})], "
