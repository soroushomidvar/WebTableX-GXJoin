from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class SplitG (RawPatternBlock):
    NAME = "SPLITG"

    def __init__(self, splitter=None, index_start=None, index_end=None):
        # To init hash:
        self._hash = None
        self._splitter = None
        self._index_start = None
        self._index_end = None

        self.splitter = splitter
        self.index_start = index_start
        self.index_end = index_end


    def update_hash(self):
        self._hash = hash((self.NAME, self.index_start, self.index_end, self.splitter))

    @property
    def splitter(self):
        return self._splitter

    @splitter.setter
    def splitter(self, splitter):
        self._splitter = splitter
        self.update_hash()

    @property
    def index_start(self):
        return self._index_start

    @index_start.setter
    def index_start(self, index_start):
        self._index_start = index_start
        self.update_hash()


    @property
    def index_end(self):
        return self._index_end

    @index_end.setter
    def index_end(self, index_end):
        self._index_end = index_end
        self.update_hash()

    def apply(self, inp):
        splitted_inp = inp.split(self.splitter)
        n = len(splitted_inp)

        if self.index_start == 'e':
            self.index_start = 'e-0'
        elif self.index_start == 's':
            self.index_start = 's+0'

        if self.index_end == 'e':
            self.index_end = 'e-0'
        elif self.index_end == 's':
            self.index_end = 's+0'

        if self.index_start.startswith('e-'):
            tmp = self.index_start.split('-')
            assert len(tmp) == 2 and tmp[0] == 'e'
            real_start = n - int(tmp[1])

        elif self.index_start.startswith('s+'):
            tmp = self.index_start.split('+')
            assert len(tmp) == 2 and tmp[0] == 's'
            real_start = int(tmp[1])
        elif self.index_start == 's':
            real_start = 0
        else:
            print(f"Wrong params for start index of Split: {self.index_start}")
            raise IndexError("Wrong start")


        if self.index_end.startswith('e-'):
            tmp = self.index_end.split('-')
            assert len(tmp) == 2 and tmp[0] == 'e'
            real_end = n - int(tmp[1])
        elif self.index_end.startswith('s+'):
            tmp = self.index_end.split('+')
            assert len(tmp) == 2 and tmp[0] == 's'
            real_end = int(tmp[1])
        else:
            print(f"Wrong params for end index of Split: {self.index_end}")
            raise IndexError("Wrong end")


        try:
            exp = splitted_inp[real_start:real_end]
            return self.splitter.join(exp)

        except IndexError:
            raise IndexError("Split index not in the array")


    def get_complexity(self):
        return 3

    def get_param_count(self):
        return 3

    def get_generalizability_index(self):
        lst = [self.index_start, self.index_end]
        return sum(1 if (l == 'e-0' or l == 's+0') else 0 for l in lst)

    @classmethod
    def extract(cls, inp, blk):
        tmp = set()

        splitters = [blk.begin_sep, blk.end_sep]

        for sp in splitters:
            if sp is not None:
                parts = inp.split(sp)
                txts = blk.text.split(sp)
                n = len(txts)
                for idx, part in enumerate(parts):
                    if idx <= len(parts) - n:
                        is_ok = True
                        ttt = []
                        for i, txt in enumerate(txts):
                            ttt.append(idx + i)
                            if txt != parts[idx + i] or txt == '':
                                is_ok = False
                                break
                        if is_ok and len(ttt) > 0:
                            # Just verify #
                            s = ""
                            for i in ttt:
                                s += parts[i] + sp
                            s = s[:-1]
                            assert s == blk.text

                            st1 = f"s+{ttt[0]}"
                            st2 = f"e-{ttt[-1]}"
                            en1 = f"s+{ttt[-1] + 1}"
                            en2 = f"e-{len(parts) - 1 - ttt[-1]}"

                            tmp.add(SplitG(sp, st1, en1))
                            tmp.add(SplitG(sp, st1, en2))
                            tmp.add(SplitG(sp, st2, en1))
                            tmp.add(SplitG(sp, st2, en2))


        if blk.start > 0:
            sp = blk.text[0]
            parts = inp.split(sp)
            for idx, part in enumerate(parts):
                if sp + part == blk.text:
                    from Transformation.Blocks.LiteralPatternBlock import LiteralPatternBlock
                    tmp.add(
                        (LiteralPatternBlock(sp), SplitG(sp, f"s+{idx}", f"s+{idx + 1}"))
                    )
                    tmp.add(
                        (LiteralPatternBlock(sp), SplitG(sp, f"e-{idx}", f"e-{len(parts) -1 - idx}"))
                    )

        return tmp

    @classmethod
    def get_param_space(cls, inp_lst):
        raise NotImplementedError

    @classmethod
    def get_random(cls, inp_charset, input_max_len):
        raise NotImplementedError

    def __eq__(self, other):
        return self.splitter == other.splitter and self.index_start == other.index_start\
               and self.index_end == other.index_end

    def __hash__(self):
        return self._hash
        # return hash((self.splitter, ))

    def __repr__(self):
        return f"[SplitG: ('{self.splitter}', {self.index_start}, {self.index_end}) ], "

