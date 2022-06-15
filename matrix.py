from typing import List, Tuple, Union


class Matrix:
    # Type hints
    _data: Union[List[Union[List[Union[int, float]], Tuple[Union[int, float]]]], Tuple[
        Union[List[Union[int, float]], Tuple[Union[int, float]]]]]
    _dim: Tuple[int, int]

    def __init__(self, data):
        if not isinstance(data, (list, tuple)):
            raise TypeError("Only lists/tuples are allowed.")

        if not all([isinstance(row, (list, tuple)) for row in data]):
            raise TypeError("Only lists/tuples of lists/tuples are allowed.")

        if not all([isinstance(item, (int, float)) for row in data for item in row]):
            raise TypeError("Only integers/floats are allowed.")

        rows_len = set([len(row) for row in data])
        if len(rows_len) > 1:
            raise ValueError("Rows are not of the same size.")

        self._data = [list(i) for i in data]
        self._dim = (len(self._data), len(self._data[0]))

    def __setattr__(self, key, value):
        if self.__dict__.get(key) is None:
            super().__setattr__(key, value)
        else:
            raise AttributeError("can't set attribute")

    def __repr__(self):
        return str(self._data)

    def __str__(self):
        return str(self._data)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            c = [[item + other for item in row] for row in self._data]
        elif isinstance(other, Matrix):
            if self._dim != other._dim:
                raise ValueError(f"operands could not be broadcast together with shapes {self._dim} {other._dim}")
            else:
                c = [[self._data[i][j] + other._data[i][j] for j in range(len(self._data[i]))] for i in
                     range(len(self._data))]
        else:
            raise TypeError("Only integers/floats/Matrix types are allowed.")
        return Matrix(c)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            c = [[item - other for item in row] for row in self._data]
        elif isinstance(other, Matrix):
            if self._dim != other._dim:
                raise ValueError(f"operands could not be broadcast together with shapes {self._dim} {other._dim}")
            else:
                c = [[self._data[i][j] - other._data[i][j] for j in range(len(self._data[i]))] for i in
                     range(len(self._data))]
        else:
            raise TypeError("Only integers/floats/Matrix types are allowed.")
        return Matrix(c)

    def __rsub__(self, other):
        c = self.__mul__(-1)
        return c.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            c = [[item * other for item in row] for row in self._data]
        elif isinstance(other, Matrix):
            if self._dim[1] == other._dim[0]:
                c = []
                for i in range(self._dim[0]):
                    tmp = []
                    row = self._data[i]
                    for j in range(other._dim[1]):
                        product = sum([row[k] * other._data[k][j] for k in range(self._dim[1])])
                        tmp.append(product)
                    c.append(tmp)
            else:
                raise ValueError(f"operands could not be broadcast together with shapes {self._dim} {other._dim}")
        else:
            raise TypeError("Only integers/floats/Matrix types are allowed.")
        return Matrix(c)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, key):
        return self._data[key]

    def transpose(self):
        t = []
        for column in range(self._dim[1]):
            tmp = []
            for row in range(self._dim[0]):
                tmp.append(self._data[row][column])
            t.append(tmp)
        return Matrix(t)

    def _sub_matrices(self):
        subs = []
        # loop over the elements of the first row
        for j in range(self._dim[1]):
            tmp = []
            # loop through the all the rows except the first one
            for i in range(1, self._dim[0]):
                # ignore the column where the element exists
                sub_row = [self._data[i][z] for z in range(self._dim[1]) if z != j]
                tmp.append(sub_row)
            subs.append(Matrix(tmp))
        return subs

    # complexity of n!/2 recursive implementation
    def det(self):
        if self._dim[0] != self._dim[1]:
            raise ValueError(f"Operation is valid only on square matrix")
        if self._dim[0] == 2:
            return self._data[0][0] * self._data[1][1] - self._data[0][1] * self._data[1][0]
        else:
            subs = self._sub_matrices()
            det = [-1 * self._data[0][i] * subs[i].det() if i % 2 else self._data[0][i] * subs[i].det() for i in
                   range(len(subs))]
            return sum(det)
