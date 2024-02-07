import re

from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class SplitSubstrG (RawPatternBlock):
    NAME = "SPLT_SUB_G"

    def __init__(self, splitter=None, index=None, start=None, end=None):
        # To init hash:
        self._hash = None
        self._splitter = None
        self._index = None
        self._start = None
        self._end = None

        self.splitter = splitter
        self.index = index
        self.start = start
        self.end = end


    def update_hash(self):
        self._hash = hash((self.NAME, self.start, self.end, self.index, self.splitter))

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

    @property
    def splitter(self):
        return self._splitter

    @splitter.setter
    def splitter(self, splitter):
        self._splitter = splitter
        self.update_hash()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index
        self.update_hash()

    def apply(self, inp):
        splitted_inp = inp.split(self.splitter)
        n = len(splitted_inp)
        try:

            if self.start == 'e':
                self.start = 'e-0'
            elif self.start == 's':
                self.start = 's+0'

            if self.end == 'e':
                self.end = 'e-0'
            elif self.end == 's':
                self.end = 's+0'

            if self.index == 'e':
                self.index = 'e-0'
            elif self.index == 's':
                self.index = 's+0'

            # Split
            if self.index.startswith('e-'):
                tmp = self.index.split('-')
                assert len(tmp) == 2 and tmp[0] == 'e'
                real_index = n - int(tmp[1])
            elif self.index.startswith('s+'):
                tmp = self.index.split('+')
                assert len(tmp) == 2 and tmp[0] == 's'
                real_index = int(tmp[1])
            elif self.index == 's':
                real_index = 0
            else:
                print(f"Wrong params for SplitSubstr: {self.index}")
                raise IndexError("Wrong index")

            s = splitted_inp[real_index]



            # Substring
            n = len(s)
            if self.start.startswith('e-'):
                tmp = self.start.split('-')
                assert len(tmp) == 2 and tmp[0] == 'e'
                real_start = n - int(tmp[1])

            elif self.start.startswith('s+'):
                tmp = self.start.split('+')
                assert len(tmp) == 2 and tmp[0] == 's'
                real_start = int(tmp[1])
            elif self.start == 's':
                real_start = 0
            else:
                print(f"Wrong params for SplitSubstr: {self.start}")
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
                print(f"Wrong params for SplitSubstr end: {self.end}")
                raise IndexError("Wrong end")

            if real_end > len(s):
                real_end = len(s)

            return s[real_start:real_end]

        except IndexError:
            raise IndexError("Split index not in the array")


    def get_complexity(self):
        return 4


    def get_param_count(self):
        return 4

    def get_generalizability_index(self):
        lst = [self.index, self.start, self.end]
        return sum(1 if (l == 'e-0' or l == 's+0') else 0 for l in lst)

    @classmethod
    def get_param_space(cls, inp_lst):
        raise NotImplementedError

    @classmethod
    def get_random(cls, inp_charset, input_max_len):
        raise NotImplementedError


    @classmethod
    def extract(cls, inp, blk):

        tmp = set()
        out = blk.text

        chars = {c for c in inp}
        out_chars = {c for c in out}
        chars = chars - out_chars

        n = len(out)

        for ch in chars:
            spt = inp.split(ch)
            for idx, sp in enumerate(spt):
                if len(sp) > n:  # not >= because it will be same as split
                    matches = [m.start() for m in re.finditer('(?='+re.escape(out)+')', sp)]
                    for m in matches:
                        end = m+n
                        idx1 = f"s+{idx}"
                        idx2 = f"e-{len(spt)-1-idx}"
                        st1 = f"e-{len(sp) - m}"
                        st2 = f"s+{m}"
                        en1 = f"e-{len(sp) - end}"
                        en2 = f"s+{end}"

                        tmp.add(SplitSubstrG(ch, idx1, st1, en1))
                        tmp.add(SplitSubstrG(ch, idx1, st1, en2))
                        tmp.add(SplitSubstrG(ch, idx1, st2, en1))
                        tmp.add(SplitSubstrG(ch, idx1, st2, en2))

                        tmp.add(SplitSubstrG(ch, idx2, st1, en1))
                        tmp.add(SplitSubstrG(ch, idx2, st1, en2))
                        tmp.add(SplitSubstrG(ch, idx2, st2, en1))
                        tmp.add(SplitSubstrG(ch, idx2, st2, en2))


        return tmp


    def __eq__(self, other):
        return self.splitter == other.splitter and self.index == other.index \
               and self.start == other.start and self.end == other.end

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"[SplitSubstrG: '{self.splitter}', {self.index}, ({self.start},{self.end}) ], "

