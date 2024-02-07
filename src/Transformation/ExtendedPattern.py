from Transformation.Pattern import Pattern


class ExtendedPattern:
    def __init__(self, transformations=[], original_tr=None):
        self._hash = None
        self.transformations = transformations
        self.original_transformation = original_tr


    def update_hash(self):
        self._hash = hash(tuple([b.__hash__() for b in self.transformations]))

    @property
    def transformations(self):
        return self._transformations

    @transformations.setter
    def transformations(self, inpt):
        if type(inpt) not in [list, tuple]:
            raise ValueError('blocks must be a list')

        transformations = []
        for inp in inpt:
            if type(inp) != Pattern:   # is a subclass of Pattern
                raise ValueError('input type not Pattern')
            transformations.append(inp)


        self._transformations = transformations
        self.update_hash()


    def __hash__(self):
        return self._hash


    def apply(self, inp):
        s = set()
        for b in self.transformations:
            try:
                s.add(b.apply(inp))
            except IndexError:
                pass
        return s



    def get_complexity(self):
        return self.original_transformation.get_complexity()

    def get_param_count(self):
        return self.original_transformation.get_param_count()

    def get_generalizability_index(self):
        return self.original_transformation.get_generalizability_index()


    def __len__(self):
        return len(self.original_transformation)


    def __repr__(self):
        return f"Gen {self.original_transformation}"
    
