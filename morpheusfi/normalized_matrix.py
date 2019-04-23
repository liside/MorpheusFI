# Copyright 2019 Side Li, Lingjiao Chen and Arun Kumar
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.sparse as sp
from numpy.core.numeric import isscalar
from numpy.matrixlib.defmatrix import asmatrix, matrix
import numpy.core.numeric as N
import time
import comp
import base

class NormalizedMatrix(matrix):
    __array_priority__ = 12.0

    def __new__(cls, ent_table, att_table, kfkds,
                dtype=None, copy=False, trans=False, stamp=None, second_order=False,
                shadow_ent_table=None, shadow_att_table=None,
                a=1.0, b=0.0, c=1.0, identity=False):
        """
        Matrix constructor
        
        Parameters
        ---------
        ent_table: numpy matrix
        att_table: list of numpy matrix
        kfkds: list of numpy array
        dtype: data type
        copy: whether to copy
        trans: boolean, indicating whether the matrix is transposed
        stamp: time stamp on normalized matrix

        Examples
        --------
        Entity Table:
            matrix([[ 1.  2.]
                    [ 4.  3.]
                    [ 5.  6.]
                    [ 8.  7.]
                    [ 9.  1.]])
        List of Attribute Table:
            [
                matrix([[ 1.1  2.2]
                        [ 3.3  4.4]])
            ]
        K:
            [
                array([0, 1, 1, 0, 1])
             ]
        Transposed:
            False
        """
        obj = N.ndarray.__new__(NormalizedMatrix, dtype)
        sizes = [ent_table.shape[1]] + [t.shape[1] for t in att_table]
        if second_order and copy is False:
            obj.ent_table, obj.att_table, obj.kfkds = NormalizedMatrix.data_interaction(ent_table, kfkds, att_table)
            obj.shadow_att_table = att_table
            obj.shadow_ent_table = (None if ent_table.size == 0 else ent_table)
            d = (3 + sum(sizes)) * sum(sizes) / 2
            obj.nshape = (d, len(kfkds[0])) if trans else (len(kfkds[0]), d)
        else:
            obj.ent_table = ent_table
            obj.att_table = att_table
            obj.kfkds = kfkds
            obj.shadow_att_table = shadow_att_table
            obj.shadow_ent_table = shadow_ent_table
            obj.nshape = (sum(sizes), len(kfkds[0])) if trans else (len(kfkds[0]), sum(sizes))

        if second_order and copy:
            d = sum(sizes)
            for i in range(len(att_table)):
                for j in range(i + 1, len(att_table)):
                    d += shadow_att_table[i].shape[1] * shadow_att_table[j].shape[1]
            if ent_table.size > 0:
                for i in range(len(att_table)):
                    d += shadow_ent_table.shape[1] * shadow_att_table[i].shape[1]
            obj.nshape = (d, len(kfkds[0])) if trans else (len(kfkds[0]), d)

        if identity:
            obj.nshape = (obj.nshape[1] + sum([t.shape[1] for t in att_table]), obj.nshape[0]) if trans else \
                (obj.nshape[0], obj.nshape[1] + sum([t.shape[0] for t in att_table]))

        obj.trans = trans
        obj.second_order = second_order
        obj.stamp = time.clock() if stamp is None else stamp
        obj.sizes = sizes
        obj.a = a
        obj.b = b
        obj.c = c
        obj.identity = identity
        # used for future operators
        obj.indexes = reduce(lambda x, y: x + [(x[-1][1], x[-1][1] + y)], sizes, [(0, 0)])[2:]

        return obj

    def _copy(self, ent_table, att_table, a=None, b=None, c=None, shadow_att=None, shadow_ent=None, identity=None):
        """
        Copy constructor
        """
        return NormalizedMatrix(ent_table, att_table,
                                self.kfkds, copy=True,
                                dtype=self.dtype, trans=self.trans, second_order=self.second_order,
                                a=(self.a if a is None else a), b=(self.b if b is None else b),
                                c=(self.c if c is None else c),
                                shadow_att_table=(self.shadow_att_table if shadow_att is None else shadow_att),
                                shadow_ent_table=(self.shadow_ent_table if shadow_ent is None else shadow_ent),
                                identity=(self.identity if identity is None else identity))

    def __getitem__(self, index):
        """
        Slicing is not supported. It will cause significant performance penalty on
        normalized matrix.

        :param index: slicing index
        :return: error
        """
        return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "\n".join(["Entity Table:", self.ent_table.__str__(),
              "Attribute Table:", "\n".join((t.__str__() for t in self.att_table)),
              "K matrix:", "\n".join((t.__str__() for t in self.kfkds)),
              "Transposed:", self.trans.__str__()])

    """
    Array functions are created to follow numpy semantics.
    https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
    
    1. Explicit constructor call
    2. View casting
    3. Creating from new template
    """
    def __array_prepare__(self, obj, context=None):
        pass

    def __array_wrap__(self, out_arr, context=None):
        pass

    def __array_finalize__(self, obj):
        pass

    _SUPPORTED_UFUNCS = {np.add: {1: "__add__", -1: "__radd__"},
                         np.subtract: {1: "__sub__", -1: "__rsub__"},
                         np.divide: {1: "__div__", -1: "__rdiv__"},
                         np.multiply: {1: "__mul__", -1: "__rmul__"},
                         np.power: {1: "__pow__", -1: "__rpow__"}}

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle ufunc supported in numpy standard library.
        reference: https://docs.scipy.org/doc/numpy-1.13.0/reference/ufuncs.html

        :param ufunc: ufunc object
        :param method: type of method. In this class, only __call__ is handled
        :param inputs:
        :param kwargs:
        :return: Normalized matrix or matrix or ndarray or numeric
        """
        if ufunc in self._SUPPORTED_UFUNCS and len(inputs) == 2 and method == "__call__":
            order = isinstance(inputs[0], NormalizedMatrix) - isinstance(inputs[1], NormalizedMatrix)
            if order == 1:
                return getattr(inputs[0], self._SUPPORTED_UFUNCS[ufunc][order])(inputs[1], **kwargs)
            if order == -1:
                return getattr(inputs[1], self._SUPPORTED_UFUNCS[ufunc][order])(inputs[0], **kwargs)
            if order == 0 and ufunc is np.multiply:
                return inputs[0].__mul__(inputs[1], **kwargs)

        return NotImplemented

    # Element-wise Scalar Operators
    # Lazy evaluations
    def __add__(self, other):
        if isscalar(other):
            return self._copy(self.ent_table, self.att_table, b=other+self.b)

        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isscalar(other):
            self.b = self.b + other
            return other

        return NotImplemented

    def __sub__(self, other):
        if isscalar(other):
            return self._copy(self.ent_table, self.att_table, b=self.b-other)

        return NotImplemented

    def __rsub__(self, other):
        if isscalar(other):
            return self._copy(self.ent_table, self.att_table, a=-self.a, b=other-self.b)

        return NotImplemented

    def __isub__(self, other):
        if isscalar(other):
            self.b = self.b - other
            return self

    def __mul__(self, other):
        if isinstance(other, NormalizedMatrix):
            if self.stamp == other.stamp and self.trans ^ other.trans:
                return self._cross_prod()
            else:
                return NotImplemented

        if isinstance(other, (N.ndarray, list, tuple)):
            # This promotes 1-D vectors to row vectors
            if self.trans:
                return self._right_matrix_multiplication(self, asmatrix(other).T).T
            else:
                return self._left_matrix_multiplication(self, asmatrix(other))

        if isscalar(other) or not hasattr(other, '__rmul__'):
            return self._copy(self.ent_table, self.att_table, a=self.a*other)

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (N.ndarray, list, tuple)):
            if self.trans:
                return self._left_matrix_multiplication(self, asmatrix(other).T).T
            else:
                return self._right_matrix_multiplication(self, asmatrix(other))

        if isscalar(other) or not hasattr(other, '__rmul__'):
            return self._copy(self.ent_table, self.att_table, a=self.a*other)

        return NotImplemented

    def __imul__(self, other):
        if not isscalar(other):
            return NotImplemented

        self.a = self.a * other
        return self

    def __div__(self, other):
        if isscalar(other):
            return self._copy(self.ent_table, self.att_table, a=self.a/other)

        return NotImplemented

    def __rdiv__(self, other):
        if isscalar(other):
            return self._copy(self.ent_table, self.att_table, a=other/self.a, c=-self.c)

        return NotImplemented

    def __idiv__(self, other):
        if isscalar(other):
            self.a = self.a / other
            return self

        return NotImplemented

    def __pow__(self, other):
        if not isscalar(other) or self.b != 0.0:
            return NotImplemented

        return self._copy(
            (self.ent_table.power(other) if sp.issparse(self.ent_table) else np.power(self.ent_table, other)),
            [(t.power(other) if sp.issparse(t) else np.power(t, other)) for t in self.att_table],
            a=self.a**other, c=self.c*other,
            shadow_att=([(t.power(other) if sp.issparse(t) else np.power(t, other)) for t in self.shadow_att_table]
                        if self.shadow_att_table is not None else None),
            shadow_ent=((self.shadow_ent_table.power(other) if sp.issparse(self.shadow_ent_table)
                         else np.power(self.shadow_ent_table, other)) if self.shadow_ent_table is not None else None))

    def __rpow__(self, other):
        self.__pow__(other)

    def __ipow__(self, other):
        if not isscalar(other) or self.b != 0.0:
            return NotImplemented
        self.c = self.c * other
        self.a = self.a ** other
        self.ent_table = self.ent_table.power(2) if sp.issparse(self.ent_table) else np.power(self.ent_table, other)
        self.att_table = [(t.power(other) if sp.issparse(t) else np.power(t, other)) for t in self.att_table]
        if self.shadow_att_table is not None:
            self.shadow_att_table = [(t.power(other) if sp.issparse(t) else np.power(t, other))
                                     for t in self.shadow_att_table]
        if self.shadow_ent_table is not None:
            self.shadow_ent_table = (self.shadow_ent_table.power(other) if self.shadow_ent_table is not None else None)

        return self

    # Aggregation
    def sum(self, axis=None, dtype=None, out=None):
        """
        Paramters
        ---------
        axis: None or int or tuple of ints, optional
            the axis used to perform sum aggreation.

        Examples
        --------
        T = Entity Table:
                [[ 1.  2.]
                 [ 4.  3.]
                 [ 5.  6.]
                 [ 8.  7.]
                 [ 9.  1.]]
            Attribute Table:
                [[ 1.1  2.2]
                 [ 3.3  4.4]]
            K:
                [[1, 0, 0, 1, 0]]
        >>> T.sum(axis=0)
            [[ 27.   19.   12.1  17.6]]
        >>> T.sum(axis=1)
            [[  6.3]
             [ 14.7]
             [ 18.7]
             [ 18.3]
             [ 17.7]]
        >>> T.sum()
            75.7
        """
        k = self.kfkds
        ns = k[0].shape[0]
        nr = [t.shape[0] for t in self.att_table]
        if axis == 0:
            # col sum
            if self.trans:
                rr = []
                if self.second_order:
                    r_sum = [t.sum(axis=1)[self.kfkds[i]] for i, t in enumerate(self.shadow_att_table)]
                    for i in range(len(r_sum)):
                        for j in range(i + 1, len(r_sum)):
                            rr += [np.multiply(r_sum[i], r_sum[j])]

                res = self.ent_table.sum(axis=1).reshape((ns, -1)) + \
                      sum((t.sum(axis=1)[self.kfkds[i]] for i, t in enumerate(self.att_table))).reshape((ns, -1)) \
                      + sum(rr)

                if self.identity:
                    res += ns * len(self.att_table)
                if self.a != 1.0:
                    res *= self.a
                if self.b != 0.0:
                    res += self.shape[1] * self.b

                return res.T
            else:
                other = np.ones((1, ns))

                return self._right_matrix_multiplication(self, other)
        elif axis == 1:
            # row sum
            if self.trans:
                other = np.ones((1, ns))

                return self._right_matrix_multiplication(self, other).T
            else:
                rr = self._row_sum_rr()
                res = sum((t.sum(axis=1)[self.kfkds[i]] for i, t in enumerate(self.att_table))).reshape((ns, -1)) \
                       + sum(rr)

                if self.ent_table is not None and self.ent_table.size > 0:
                    res += self.ent_table.sum(axis=1).reshape((ns, -1))

                if self.identity:
                    res += 1 * len(self.att_table)
                if self.a != 1.0:
                    res *= self.a
                if self.b != 0.0:
                    res += self.shape[1] * self.b

                return res

        # sum of the whole matrix
        # res is k * r
        other = np.ones((1, ns))
        g = [np.zeros((1, t.shape[0]), dtype=np.float64) for t in self.att_table]
        comp.group(ns, len(k), 1, k, nr, other, g)
        res = self.ent_table.sum() + \
            sum((g[i] * np.matrix(t.sum(axis=1).reshape(-1, 1)) for i, t in enumerate(self.att_table)))._collapse(None)

        if self.second_order:
            res += sum(self._row_sum_rr()).sum()

        if self.identity:
            res += ns * len(self.att_table)

        if self.a != 1.0:
            res = res * self.a
        if self.b != 0.0:
            res = res + self.shape[1] * self.shape[0] * self.b

        return res

    def _row_sum_rr(self):
        rr = []
        if self.second_order:
            r_sum = [t.sum(axis=1)[self.kfkds[i]] for i, t in enumerate(self.shadow_att_table)]
            for i in range(len(r_sum)):
                for j in range(i + 1, len(r_sum)):
                    rr += [np.multiply(r_sum[i], r_sum[j])]

            if self.shadow_ent_table is not None:
                for i in range(len(r_sum)):
                    rr += [np.multiply(r_sum[i], self.shadow_ent_table.sum(axis=1))]
        return rr

    # Multiplication
    def _left_matrix_multiplication(self, n_matrix, other, permute=None):
        s = n_matrix.ent_table
        k = n_matrix.kfkds
        r = n_matrix.att_table

        ns = k[0].shape[0]
        ds = s.shape[1]
        dw = other.shape[1]

        if s.size > 0:
            res = np.asfortranarray(s.dot(other[0:ds]), dtype=float)
        else:
            res = np.zeros((ns, dw), dtype=np.float64, order='F')

        v_list = []
        start = ds
        if self.identity:
            for i in range(len(k)):
                r_buff = r[i]
                nr, dr = r_buff.shape[0], r_buff.shape[1]
                # identity part
                v_list.append(np.asfortranarray(other[start:start + nr]))
                # normal
                end = start + nr + dr
                v = r_buff.dot(other[start + nr: end])
                start += nr + dr
                v_list.append(np.asfortranarray(v))
        else:
            for i in range(len(k)):
                r_buff = r[i]
                nr, dr = r_buff.shape[0], r_buff.shape[1]
                end = start + dr
                v = r_buff.dot(other[start:end])
                start += dr
                v_list.append(np.asfortranarray(v, dtype=float))

        comp.add_new(self.identity, ns, len(k), dw, k, v_list, [v.shape[0] for v in v_list], res)

        if self.second_order:
            # sr
            if self.shadow_ent_table is not None:
                for i in range(len(r)):
                    # sr
                    ss = self.shadow_ent_table
                    r1 = self.shadow_att_table[i]
                    k1 = self.kfkds[i]
                    dr1 = r1.shape[1]
                    dss = ss.shape[1]

                    if not sp.issparse(r1) and not sp.issparse(ss):
                        if permute is None:
                            kr1 = r1.dot(other[start:start + dr1 * dss].reshape((dr1, -1), order='F'))
                        else:
                            kr1 = r1.dot(other[start:start + dr1 * dss][permute].reshape((dr1, -1), order='F'))

                        comp.hadamard_rowsum(ns, dss, other.shape[1], True,
                                             self.kfkds[i], self.kfkds[i], kr1, ss, np.asfortranarray(res))
                    else:
                        kr1 = r1 * other[start:start + dr1 * dss].reshape((dr1, -1), order='F')

                        comp.hadamard_rowsum_sparse(ns, len(ss.data), kr1.shape[0], r1.shape[0], dss, other.shape[1],
                                                    True, k1, k1, kr1,
                                                    ss.row, ss.col, ss.data,  np.asfortranarray(res))
                    start += dr1 * dss

            # rr
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    r1, r2 = self.shadow_att_table[i], self.shadow_att_table[j]
                    k1, k2 = self.kfkds[i], self.kfkds[j]
                    dr1, dr2 = r1.shape[1], r2.shape[1]

                    if not sp.issparse(r1) and not sp.issparse(r2):
                        kr1 = r1.dot(other[start:start + dr1 * dr2].reshape((dr1, -1), order='F'))
                        comp.hadamard_rowsum(ns, dr2, other.shape[1], False,
                                             self.kfkds[i], self.kfkds[j], kr1, r2, np.asfortranarray(res))
                    else:
                        kr1 = r1 * other[start:start + dr1 * dr2].reshape((dr1, -1), order='F')
                        comp.hadamard_rowsum_sparse(ns, len(r2.data), r1.shape[0], r2.shape[0], dr2, other.shape[1],
                                                    False, k1, k2, kr1,
                                                    r2.row, r2.col, r2.data,  np.asfortranarray(res))
                    start += dr1 * dr2
        # lazy evaluation of scalar operators
        if self.a != 1.0:
            res = res * self.a
        if self.b != 0.0:
            res = res + other.sum(axis=0) * self.b

        return res

    def _right_matrix_multiplication(self, n_matrix, other, permute=None):
        other = other.astype(np.float, order='C')
        s = n_matrix.ent_table
        k = n_matrix.kfkds
        r = n_matrix.att_table

        ns = k[0].shape[0]
        nr = [t.shape[0] for t in r]
        nk = len(k)
        nw = other.shape[0]
        res = [np.zeros((nw, t.shape[0]), dtype=float) for t in r]

        comp.group(ns, nk, nw, k, nr, other, res)
        rr = []
        if self.second_order:
            if self.shadow_ent_table is not None:
                for i in range(len(r)):
                    # sr
                    ss = self.shadow_ent_table
                    r1 = self.shadow_att_table[i]
                    k1 = self.kfkds[i]

                    # TODO: 1. avoid materializing a map 2. use 2d array in transformation
                    if not sp.issparse(s) and not sp.issparse(r1):
                        u = np.zeros((r1.shape[0], ss.shape[1] * nw), dtype=float, order='C')
                        comp.group_left(ss.shape[0], ss.shape[1] * nw,
                                        np.ascontiguousarray(base.data_interaction_rr(ss, np.matrix(other).T)), k1, u)
                        if permute is None:
                            rr += [np.matrix(base.data_interaction_rr(r1, u).sum(axis=0).reshape(nw, -1))]
                        else:
                            rr += [np.matrix(base.data_interaction_rr(r1, u).sum(axis=0)[permute].reshape(nw, -1))]
                    else:
                        u = np.zeros((nw, r1.shape[1] * ss.shape[1]), dtype=float, order='C')
                        comp.data_interaction_group_sparse(ns, len(ss.data), ss.shape[0], ss.shape[1],
                                                           len(r1.data), r1.shape[0], r1.shape[1], nw,
                                                           ss.row, ss.col, ss.data, r1.row, r1.col, r1.data,
                                                           k1, k1, np.ascontiguousarray(other), u)

            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    r1, r2 = self.shadow_att_table[i], self.shadow_att_table[j]
                    k1, k2 = self.kfkds[i], self.kfkds[j]

                    r1, r2 = r2, r1
                    k1, k2 = k2, k1

                    if not sp.issparse(r1) and not sp.issparse(r2):
                        u = np.zeros((r2.shape[0], r1.shape[1] * nw), dtype=float, order='C')
                        comp.data_interaction_group(ns, r1.shape[1], nw, k1, k2, r1, np.ascontiguousarray(other), u)
                        rr += [np.matrix(base.data_interaction_rr(r2, u).sum(axis=0).reshape(nw, -1))]
                    else:
                        u = np.zeros((nw, r1.shape[1] * r2.shape[1]), dtype=float, order='C')
                        r1 = r1.tocsr()[k1].tocoo()
                        comp.data_interaction_group_sparse(ns, len(r1.data), r1.shape[0], r1.shape[1],
                                                           len(r2.data), r2.shape[0], r2.shape[1], nw,
                                                           r1.row, r1.col, r1.data, r2.row, r2.col, r2.data,
                                                           k1, k2, np.ascontiguousarray(other), u)
                        rr += [u]

        if self.identity:
            l = ([other * s if sp.issparse(s) else other.dot(s)] if s.shape[1] != 0 else [])
            for i, t in enumerate(r):
                l.append(np.matrix(res[i]))
                l.append(res[i] * t if sp.issparse(t) else res[i].dot(t))
            l += rr
            total = np.hstack(l)
        else:
            total = np.hstack(([other * s if sp.issparse(s) else other.dot(s)] if s.shape[1] != 0 else []) +
                             [(res[i] * t if sp.issparse(t) else res[i].dot(t)) for i, t in enumerate(r)] + rr)

        if self.a != 1.0:
            total = total * self.a
        if self.b != 0.0:
            total = total + self.b * other.sum(axis=1)

        return total

    def _cross_prod(self):
        s = self.ent_table
        r = self.att_table
        k = self.kfkds
        ns = k[0].shape[0]
        ds = s.shape[1]
        nr = [t.shape[0] for t in self.att_table]
        dr = [t.shape[1] for t in self.att_table]

        if not self.trans:
            if all(map(sp.issparse, self.att_table)):
                return NotImplemented
            else:
                if s.size > 0:
                    res = self._t_cross(s)
                else:
                    res = np.zeros((ns, ns), dtype=float, order='C')

                if all(map(sp.issparse, r)):
                    cross_r = [self._t_cross(t).toarray() for t in r]
                else:
                    cross_r = [self._t_cross(t) for t in r]
                comp.expand_add(ns, len(k), k, cross_r, nr, res)

                return res

        else:
            if all(map(sp.issparse, self.att_table)):
                other = np.ones((1, ns))
                v = [np.zeros((1, t.shape[0]), dtype=float) for t in self.att_table]
                comp.group(ns, len(k), 1, k, nr, other, v)
                size = self.att_table[0].size
                data = np.empty(size)

                # part 2 and 3 are p.T and p
                comp.multiply_sparse(size, self.att_table[0].row, self.att_table[0].data, np.sqrt(v[0]), data)
                diag_part = self._cross(sp.coo_matrix((data, (self.att_table[0].row, self.att_table[0].col))))
                if ds > 0:
                    m = np.zeros((nr[0], ds))
                    comp.group_left(ns, ds, s, k[0], m)
                    p = self._cross(self.att_table[0], m)
                    s_part = self._cross(self.ent_table)

                    res = sp.vstack((np.hstack((s_part, p.T)), sp.hstack((p, diag_part))))
                else:
                    res = diag_part

                # multi-table join
                for i in range(1, len(k)):
                    ps = []
                    if ds > 0:
                        m = np.zeros((nr[i], ds))
                        comp.group_left(ns, ds, s, k[i], m)
                        ps += [self._cross(self.att_table[i], m)]

                    # cp (KRi)
                    size = self.att_table[i].size
                    data = np.empty(size)
                    comp.multiply_sparse(size, self.att_table[i].row, self.att_table[i].data, np.sqrt(v[i]), data)
                    diag_part = self._cross(sp.coo_matrix((data, (self.att_table[i].row, self.att_table[i].col))))

                    for j in range(i):
                        ps += [r[i].tocsr()[k[i]].T.dot(r[j].tocsr()[k[j]])]

                    res = sp.vstack((sp.hstack((res, sp.vstack([p.T for p in ps]))), sp.hstack(ps + [diag_part])))
            else:
                s = np.ascontiguousarray(s)
                if self.second_order:
                    nt = self.shape[0]
                    other = np.ones((1, ns))
                    v = [np.zeros((1, t.shape[0]), dtype=float) for t in r]

                    res = np.empty((nt, nt))
                    comp.group(ns, len(k), 1, k, nr, other, v)
                    data = sp.diags(np.sqrt(v[0]), [0]) * r[0]
                    res[ds:ds+dr[0], ds:ds+dr[0]] = self._cross(data)

                    if ds > 0:
                        # p1
                        m1 = np.zeros((nr[0], ds), dtype=float)
                        comp.group_left(ns, ds, s, k[0], m1)
                        res[ds:ds + dr[0], :ds] = r[0].T.dot(m1)
                        res[:ds, ds:ds + dr[0]] = res[ds:ds + dr[0], :ds].T
                        # s
                        res[:ds, :ds] = self._cross(self.ent_table)

                        r_p = base.data_interaction_rr(self.shadow_ent_table, self.shadow_att_table[0][k[0]])
                        # p2
                        res[:ds, ds+dr[0]:] = s.T.dot(r_p)
                        res[ds+dr[0]:, :ds] = res[:ds, ds+dr[0]:].T
                        # p3
                        m = np.zeros((nr[0], self.shadow_ent_table.shape[1]), dtype=float)
                        comp.group_left(ns, self.shadow_ent_table.shape[1], self.shadow_ent_table, k[0], m)
                        ksr = base.data_interaction_rr(m, self.shadow_att_table[0])
                        res[ds:ds+dr[0], ds+dr[0]:] = self._cross(r[0], ksr)
                        res[ds+dr[0]:, ds:ds+dr[0]] = res[ds:ds+dr[0], ds+dr[0]:].T
                        rrx = base.data_interaction_rr(self.shadow_att_table[0], self.shadow_att_table[0])
                        ssx = base.data_interaction_rr(self.shadow_ent_table, self.shadow_ent_table)
                        dss = self.shadow_ent_table.shape[1] * self.shadow_ent_table.shape[1]
                        kss = np.zeros((nr[0], dss), dtype=float)
                        comp.group_left(ns, dss, ssx, k[0], kss)
                        res[ds + dr[0]:, ds + dr[0]:] = rrx.T.dot(kss).reshape((r_p.shape[1], r_p.shape[1]))
                else:
                    nt = self.ent_table.shape[1] + self.att_table[0].shape[1]
                    other = np.ones((1, ns))
                    v = [np.zeros((1, t.shape[0]), dtype=float) for t in self.att_table]
                    comp.group(ns, len(k), 1, k, nr, other, v)

                    res = np.empty((nt, nt))
                    data = sp.diags(np.sqrt(v[0]), [0]) * r[0]
                    res[ds:, ds:] = self._cross(data)
                    if ds > 0:
                        m = np.zeros((nr[0], ds))
                        comp.group_left(ns, ds, s, k[0], m)
                        res[ds:, :ds] = self._cross(self.att_table[0], m)
                        res[:ds, ds:] = res[ds:, :ds].T
                        res[:ds, :ds] = self._cross(self.ent_table)

                    # multi-table join
                    for i in range(1, len(self.kfkds)):
                        if ds > 0:
                            m = np.zeros((nr[i], ds))
                            comp.group_left(ns, ds, s, k[i], m)
                            ni1 = ds + sum([t.shape[1] for t in self.att_table[:i]])
                            ni2 = ni1 + self.att_table[i].shape[1]
                            res[ni1:ni2, :ds] = self._cross(self.att_table[i], m)
                            res[:ds, ni1:ni2] = res[ni1:ni2, :ds].T

                        # cp(KRi)
                        data = np.empty(self.att_table[i].shape, order='C')
                        comp.multiply(self.att_table[i].shape[0], self.att_table[i].shape[1], self.att_table[i], v[i],
                                      data)
                        res[ni1:ni2, ni1:ni2] = self._cross(data)

                        for j in range(i):
                            dj1 = ds + sum([t.shape[1] for t in self.att_table[:j]])
                            dj2 = dj1 + self.att_table[j].shape[1]

                            if (ns * 1.0 / nr[j]) > (1 + nr[j] * 1.0 / dr[j]):
                                m = np.zeros((nr[i], nr[j]), order='C')
                                comp.group_k_by_k(nr[i], nr[j], ns, k[i], k[j], m)

                                res[ni1:ni2, dj1:dj2] = r[i].T.dot(m.T.dot(r[j]))
                                res[dj1:dj2, ni1:ni2] = res[ni1:ni2, dj1:dj2].T
                            else:
                                res[ni1:ni2, dj1:dj2] = r[i][k[i]].T.dot(r[j][k[j]])
                                res[dj1:dj2, ni1:ni2] = res[ni1:ni2, dj1:dj2].T

            if self.a != 1.0:
                res = res * np.power(self.a, 2)
            if self.b != 0.0:
                return NotImplemented

            return res

    def _t_cross(self, matrix_a, matrix_b=None):
        if sp.issparse(matrix_a) or sp.issparse(matrix_b):
            if matrix_b is None:
                return matrix_a * matrix_a.T
            else:
                return matrix_a * matrix_b.T
        else:
            if matrix_b is None:
                return matrix_a.dot(matrix_a.T)
            else:
                return matrix_a.dot(matrix_b.T)

    def _cross(self, matrix_a, matrix_b=None):
        if sp.issparse(matrix_a) or sp.issparse(matrix_b):
            if matrix_b is None:
                return matrix_a.T * (matrix_a)
            else:
                return matrix_a.T * (matrix_b)
        else:
            if matrix_b is None:
                return matrix_a.T.dot(matrix_a)
            else:
                return matrix_a.T.dot(matrix_b)

    def dot(self, other):
        return self.__mul__(other)

    def dot_with_permute(self, other, permute):
        return self._left_matrix_multiplication(self, other, permute)

    def right_dot_with_permute(self, other, permute):
        return self._right_matrix_multiplication(self, other, permute)

    def max(self, axis=None, out=None):
        """
        Calculate the maximum element per table or per column.
        Signatures are the same as numpy matrix to ensure downstream compatibility.

        :param axis: optional, only column wise (axis=0) operation is supported.
        :param out:
        :return: numpy matrix or numeric
        """
        if axis is None:
            return max(self.ent_table.max(), max(t.max() for t in self.att_table))

        if axis == 0:
            return np.hstack((self.ent_table.max(axis=0),
                             np.hstack([t.max(axis=0) for t in self.att_table])))

        return NotImplemented

    def min(self, axis=None, out=None):
        """
        Calculate the minmum element per table or per column.
        Signatures are the same as numpy matrix to ensure downstream compatibility.

        :param axis: optional, only column wise (axis=0) operation is supported.
        :param out:
        :return: numpy matrix or numeric
        """
        if axis is None:
            return min(self.ent_table.min(), max(t.min() for t in self.att_table))

        if axis == 0:
            return np.hstack((self.ent_table.min(axis=0),
                             np.hstack([t.min(axis=0) for t in self.att_table])))

        return NotImplemented

    def mean(self, axis=None, dtype=None, out=None):
        """
        Calculate the mean per table or per column.
        Signatures are the same as numpy matrix to ensure downstream compatibility.

        :param axis: optional, only column wise (axis=0) operation is supported.
        :param dtype: data type
        :param out:
        :return: numpy matrix or numeric
        """
        if axis is None:
            return self.sum() / (self.shape[0] * self.shape[1])

        if axis == 0:
            return np.hstack([self.ent_table.mean(axis=0)] +
                             [self.att_table[i][self.kfkds[i]].mean(axis=0) for i in range(len(self.kfkds))])


        return NotImplemented

    def transpose(self):
        return NormalizedMatrix(self.ent_table, self.att_table,
                                self.kfkds, copy=True,
                                dtype=self.dtype, trans=(not self.trans), stamp=self.stamp,
                                second_order=self.second_order,
                                a=self.a, b=self.b,
                                c=self.c,
                                shadow_att_table=self.shadow_att_table,
                                shadow_ent_table=self.shadow_ent_table,
                                identity=self.identity)
    @property
    def I(self):
        if self.trans:
            return self * self._cross_prod().I
        else:
            if self.shape[0] > self.shape[1]:
                return np.mat(self.T * self).I * self.T
            else:
                return self.T * np.mat(self * self.T).I
    @property
    def T(self):
        return self.transpose()

    @property
    def shape(self):
        return self.nshape

    @staticmethod
    def data_interaction(ent_table, kfkds, att_table):
        if any([sp.issparse(ent_table)] + map(sp.issparse, att_table)) :
            if ent_table.size > 0:
                ent_table_new = base.data_interaction(ent_table)
            else:
                ent_table_new = ent_table
        else:
            if ent_table.size > 0:
                ent_table_new = base.data_interaction(ent_table)
            else:
                ent_table_new = ent_table

        att_table_new = [base.data_interaction(t) for t in att_table]

        return ent_table_new, att_table_new, kfkds
