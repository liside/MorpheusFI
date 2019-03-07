# Copyright 2018 Side Li and Arun Kumar
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
import sklearn.preprocessing as preprocess
import scipy.sparse as sp
from numpy.testing import (
    run_module_suite, assert_equal, assert_almost_equal
)
import base
import normalized_matrix as nm


class TestNormalizedMatrix(object):
    s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
    k = [np.array([0, 1, 1, 0, 1])]

    r = [np.matrix([[1.1, 2.2], [3.3, 4.4]])]
    m = np.matrix([[1.0, 2.0, 1.1, 2.2],
                   [4.0, 3.0, 3.3, 4.4],
                   [5.0, 6.0, 3.3, 4.4],
                   [8.0, 7.0, 1.1, 2.2],
                   [9.0, 1.0, 3.3, 4.4]])
    n_matrix = nm.NormalizedMatrix(s, r, k)

    def test_add(self):
        n_matrix = self.n_matrix

        local_matrix = n_matrix + 2
        assert_equal(local_matrix.b, 2)

        local_matrix = 3 + n_matrix
        assert_equal(local_matrix.b, 3)

    def test_sub(self):
        n_matrix = self.n_matrix

        local_matrix = n_matrix - 2
        assert_equal(local_matrix.b, -2)

        local_matrix = 3 - n_matrix
        assert_equal(local_matrix.a, -1)
        assert_equal(local_matrix.b, 3)

    def test_mul(self):
        n_matrix = self.n_matrix

        local_matrix = n_matrix * 2
        assert_equal(local_matrix.a, 2)

        local_matrix = 3 * n_matrix
        assert_equal(local_matrix.a, 3)

    def test_div(self):
        n_matrix = self.n_matrix

        local_matrix = n_matrix / 2
        assert_equal(local_matrix.a, 0.5)

        local_matrix = 2 / n_matrix
        assert_equal(local_matrix.a, 2)
        assert_equal(local_matrix.c, -1)

    def test_pow(self):
        n_matrix = self.n_matrix

        local_matrix = n_matrix ** 2
        assert_equal(local_matrix.c, 2)

    def test_transpose(self):
        n_matrix = self.n_matrix
        assert_equal(n_matrix.T.T.sum(axis=0), n_matrix.sum(axis=0))
        assert_equal(np.array_equal(n_matrix.T.sum(axis=0), n_matrix.sum(axis=0)), False)

    def test_inverse(self):
        n_matrix = self.n_matrix

        assert_almost_equal(n_matrix.I, self.n_matrix.I)

    def test_row_sum(self):
        n_matrix = self.n_matrix

        assert_almost_equal(n_matrix.sum(axis=1), self.m.sum(axis=1))

        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        n_matrix = nm.NormalizedMatrix(s, r, k, second_order=True)

        rr = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(r[1][k[1]])[..., np.newaxis]).reshape(s.shape[0], -1)
        sr0 = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)
        sr1 = (np.asarray(r[1][k[1]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)

        m = np.matrix(np.hstack([n_matrix.ent_table] + [n_matrix.att_table[0][k[0]], n_matrix.att_table[1][k[1]], sr0, sr1, rr]))

        assert_almost_equal(np.sort(n_matrix.sum(axis=1), axis=0), np.sort(m.sum(axis=1), axis=0))
        assert_almost_equal(np.sort((n_matrix + 2).sum(axis=1), axis=0), np.sort((m + 2).sum(axis=1), axis=0))
        assert_almost_equal(np.sort((n_matrix * 2).sum(axis=1), axis=0), np.sort(m.dot(2).sum(axis=1), axis=0))

        # sparse
        indptr = np.array([0, 2, 3, 6, 6])
        indices = np.array([0, 2, 2, 0, 1, 4])
        data1 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        s = sp.csc_matrix((data1, indices, indptr), shape=(5, 4)).tocoo()

        row1 = np.array([0, 1, 1])
        col1 = np.array([0, 4, 1])
        data1 = np.array([1.0, 2.0, 3.0])

        row2 = np.array([1, 1, 0])
        col2 = np.array([2, 1, 1])
        data2 = np.array([1.1, 2.2, 3.3])

        k = [np.array([0, 1, 1, 0, 1]), np.array([1, 0, 0, 1, 1])]
        r = [sp.coo_matrix((data1, (row1, col1)), shape=(2, 5)),
             sp.coo_matrix((data2, (row2, col2)), shape=(2, 3))]

        n_matrix = nm.NormalizedMatrix(s, r, k, second_order=True)
        rr = (np.asarray(r[0].toarray()[k[0]][:, np.newaxis]) * np.asarray(r[1].toarray()[k[1]])[
            ..., np.newaxis]).reshape(s.shape[0], -1)

        sr0 = (np.asarray(r[0].toarray()[k[0]][:, np.newaxis]) * np.asarray(s.toarray())[..., np.newaxis]).reshape(s.shape[0], -1)
        sr1 = (np.asarray(r[1].toarray()[k[1]][:, np.newaxis]) * np.asarray(s.toarray())[..., np.newaxis]).reshape(s.shape[0], -1)

        m = np.matrix(np.hstack([n_matrix.ent_table.toarray()] + [n_matrix.att_table[0].toarray()[k[0]],
                                                        n_matrix.att_table[1].toarray()[k[1]], sr0, sr1, rr]))

        assert_almost_equal(np.sort(n_matrix.sum(axis=1), axis=0), np.sort(m.sum(axis=1), axis=0))
        assert_almost_equal(np.sort((n_matrix + 2).sum(axis=1), axis=0), np.sort((m + 2).sum(axis=1), axis=0))
        assert_almost_equal(np.sort((n_matrix * 2).sum(axis=1), axis=0), np.sort(m.dot(2).sum(axis=1), axis=0))

        # identity
        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        n_matrix = nm.NormalizedMatrix(s, r, k, identity=True)

        m = np.hstack([n_matrix.ent_table] + [np.identity(2)[k[0]], n_matrix.att_table[0][k[0]],
                                              np.identity(2)[k[1]], n_matrix.att_table[1][k[1]]])

        assert_almost_equal(np.sort(n_matrix.sum(axis=1), axis=0), np.sort(m.sum(axis=1), axis=0))
        assert_almost_equal(np.sort((n_matrix + 2).sum(axis=1), axis=0), np.sort((m + 2).sum(axis=1), axis=0))
        assert_almost_equal(np.sort((n_matrix * 2).sum(axis=1), axis=0), np.sort(m.dot(2).sum(axis=1), axis=0))

    def test_row_sum_trans(self):
        n_matrix = nm.NormalizedMatrix(self.s, self.r, self.k, trans=True)

        assert_almost_equal(n_matrix.sum(axis=1), self.m.T.sum(axis=1))

        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        n_matrix = nm.NormalizedMatrix(s, r, k, second_order=True)

        rr = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(r[1][k[1]])[..., np.newaxis]).reshape(s.shape[0], -1)
        sr0 = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)
        sr1 = (np.asarray(r[1][k[1]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)

        m = np.matrix(np.hstack([n_matrix.ent_table] + [n_matrix.att_table[0][k[0]], n_matrix.att_table[1][k[1]], sr0, sr1, rr]))

        assert_almost_equal(np.sort(n_matrix.T.sum(axis=1), axis=0), np.sort(m.T.sum(axis=1), axis=0))
        assert_almost_equal(np.sort((n_matrix + 2).T.sum(axis=1), axis=0), np.sort((m + 2).T.sum(axis=1), axis=0))
        assert_almost_equal(np.sort((n_matrix * 2).T.sum(axis=1), axis=0), np.sort(m.dot(2).T.sum(axis=1), axis=0))

    def test_col_sum(self):
        n_matrix = self.n_matrix

        assert_almost_equal(n_matrix.sum(axis=0), self.m.sum(axis=0))

        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        n_matrix = nm.NormalizedMatrix(s, r, k, second_order=True)

        rr = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(r[1][k[1]])[..., np.newaxis]).reshape(s.shape[0], -1)
        sr0 = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)
        sr1 = (np.asarray(r[1][k[1]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)

        m = np.matrix(np.hstack([n_matrix.ent_table] + [n_matrix.att_table[0][k[0]], n_matrix.att_table[1][k[1]], sr0, sr1, rr]))

        assert_almost_equal(np.sort(n_matrix.sum(axis=0)), np.sort(m.sum(axis=0)))
        assert_almost_equal(np.sort((n_matrix + 2).sum(axis=0)), np.sort((m + 2).sum(axis=0)))
        assert_almost_equal(np.sort((n_matrix * 2).sum(axis=0)), np.sort(m.dot(2).sum(axis=0)))

    def test_row_col_trans(self):
        n_matrix = nm.NormalizedMatrix(self.s, self.r, self.k, trans=True)

        assert_almost_equal(n_matrix.sum(axis=0), self.m.T.sum(axis=0))

    def test_sum(self):
        n_matrix = self.n_matrix

        assert_almost_equal(n_matrix.sum(), self.m.sum())

        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        n_matrix = nm.NormalizedMatrix(s, r, k, second_order=True)

        rr = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(r[1][k[1]])[..., np.newaxis]).reshape(s.shape[0], -1)
        sr0 = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)
        sr1 = (np.asarray(r[1][k[1]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)

        m = np.matrix(np.hstack([n_matrix.ent_table] + [n_matrix.att_table[0][k[0]], n_matrix.att_table[1][k[1]], sr0, sr1, rr]))

        assert_almost_equal(n_matrix.sum(), m.sum())
        assert_almost_equal((n_matrix+2).sum(), (m+2).sum())
        assert_almost_equal((n_matrix*2).sum(), (m*2).sum())

        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        n_matrix = nm.NormalizedMatrix(s, r, k, identity=True)

        m = np.hstack([n_matrix.ent_table] + [np.identity(2)[k[0]], n_matrix.att_table[0][k[0]],
                                              np.identity(2)[k[1]], n_matrix.att_table[1][k[1]]])

        assert_almost_equal(n_matrix.sum(), m.sum())
        assert_almost_equal((n_matrix+2).sum(), (m+2).sum())
        assert_almost_equal((n_matrix*2).sum(), (m*2).sum())

    def test_lmm(self):
        # lmm with vector
        n_matrix = self.n_matrix
        x = np.matrix([[1.0], [2.0], [3.0], [4.0]])

        assert_equal(n_matrix * x, self.m * x)

        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        x = np.matrix(np.arange(1.0, 36.0)).T
        n_matrix = nm.NormalizedMatrix(s, r, k, second_order=True)
        rr = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(r[1][k[1]])[..., np.newaxis]).reshape(s.shape[0], -1)
        sr0 = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)
        sr1 = (np.asarray(r[1][k[1]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)

        m = np.hstack([n_matrix.ent_table] + [n_matrix.att_table[0][k[0]], n_matrix.att_table[1][k[1]], sr0, sr1, rr])

        assert_almost_equal(n_matrix * x, m.dot(x))
        assert_almost_equal((n_matrix + 2) * x, (m + 2).dot(x))
        assert_almost_equal(np.power(n_matrix, 2) * x, np.power(m, 2).dot(x))

        # lmm with matrix
        x = np.hstack((np.matrix(np.arange(1.0, 36.0)).T, np.matrix(np.arange(36.0, 71.0)).T))

        assert_almost_equal(n_matrix * x, m.dot(x))
        assert_almost_equal((n_matrix + 2) * x, (m + 2).dot(x))
        assert_almost_equal(np.power(n_matrix, 2) * x, np.power(m, 2).dot(x))

        # sparse
        x = np.matrix(np.arange(1.0, 91.0)).T
        indptr = np.array([0, 2, 3, 6, 6])
        indices = np.array([0, 2, 2, 0, 1, 4])
        data1 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        s = sp.csc_matrix((data1, indices, indptr), shape=(5, 4)).tocoo()

        row1 = np.array([0, 1, 1])
        col1 = np.array([0, 4, 1])
        data1 = np.array([1.0, 2.0, 3.0])

        row2 = np.array([1, 1, 0])
        col2 = np.array([2, 1, 1])
        data2 = np.array([1.1, 2.2, 3.3])

        k = [np.array([0, 1, 1, 0, 1]), np.array([1, 0, 0, 1, 1])]
        r = [sp.coo_matrix((data1, (row1, col1)), shape=(2, 5)),
             sp.coo_matrix((data2, (row2, col2)), shape=(2, 3))]

        n_matrix = nm.NormalizedMatrix(s, r, k, second_order=True)
        rr = (np.asarray(r[0].toarray()[k[0]][:, np.newaxis]) * np.asarray(r[1].toarray()[k[1]])[
            ..., np.newaxis]).reshape(s.shape[0], -1)

        sr0 = (np.asarray(r[0].toarray()[k[0]][:, np.newaxis]) * np.asarray(s.toarray())[..., np.newaxis]).reshape(s.shape[0], -1)
        sr1 = (np.asarray(r[1].toarray()[k[1]][:, np.newaxis]) * np.asarray(s.toarray())[..., np.newaxis]).reshape(s.shape[0], -1)

        m = np.hstack([n_matrix.ent_table.toarray()] + [n_matrix.att_table[0].toarray()[k[0]],
                                                        n_matrix.att_table[1].toarray()[k[1]], sr0, sr1, rr])
        assert_almost_equal(n_matrix * x, m.dot(x))
        assert_almost_equal((n_matrix + 2) * x, (m + 2).dot(x))
        assert_almost_equal(np.power(n_matrix, 2) * x, np.power(m, 2).dot(x))

        # lmm with matrix (sparse)
        x = np.hstack((np.matrix(np.arange(1.0, 91.0)).T, np.matrix(np.arange(91, 181.0)).T))
        assert_almost_equal(n_matrix * x, m.dot(x))
        assert_almost_equal((n_matrix + 2) * x, (m + 2).dot(x))
        assert_almost_equal(np.power(n_matrix, 2) * x, np.power(m, 2).dot(x))

        # # with identity
        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        x = np.matrix(np.arange(1.0, 12.0)).T
        n_matrix = nm.NormalizedMatrix(s, r, k, identity=True)

        m = np.hstack([n_matrix.ent_table] + [np.identity(2)[k[0]], n_matrix.att_table[0][k[0]],
                                              np.identity(2)[k[1]], n_matrix.att_table[1][k[1]]])

        assert_almost_equal(n_matrix * x, m.dot(x))
        assert_almost_equal((n_matrix + 2) * x, (m + 2).dot(x))
        assert_almost_equal(np.power(n_matrix, 2) * x, np.power(m, 2).dot(x))

        # with identity and second order
        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        x = np.matrix(np.arange(1.0, 40.0)).T
        n_matrix = nm.NormalizedMatrix(s, r, k, identity=True, second_order=True)
        rr = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(r[1][k[1]])[..., np.newaxis]).reshape(s.shape[0], -1)
        sr0 = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)
        sr1 = (np.asarray(r[1][k[1]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)

        m = np.hstack([n_matrix.ent_table] + [np.identity(2)[k[0]], n_matrix.att_table[0][k[0]],
                                              np.identity(2)[k[1]], n_matrix.att_table[1][k[1]], sr0, sr1, rr])

        assert_almost_equal(n_matrix * x, m.dot(x))
        assert_almost_equal((n_matrix + 2) * x, (m + 2).dot(x))
        assert_almost_equal(np.power(n_matrix, 2) * x, np.power(m, 2).dot(x))


    def test_lmm_trans(self):
        n_matrix = self.n_matrix.T
        x = np.matrix([[1.0], [2.0], [3.0], [4.0], [5.0]])

        assert_almost_equal(n_matrix * x, self.m.T * x)

        # sparse
        x = np.matrix(np.arange(1.0, 6.0)).T
        indptr = np.array([0, 2, 3, 6, 6])
        indices = np.array([0, 2, 2, 0, 1, 4])
        data1 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        s = sp.csc_matrix((data1, indices, indptr), shape=(5, 4)).tocoo()

        row1 = np.array([0, 1, 1])
        col1 = np.array([0, 4, 1])
        data1 = np.array([1.0, 2.0, 3.0])

        row2 = np.array([1, 1, 0])
        col2 = np.array([2, 1, 1])
        data2 = np.array([1.1, 2.2, 3.3])

        k = [np.array([0, 1, 1, 0, 1]), np.array([1, 0, 0, 1, 1])]
        r = [sp.coo_matrix((data1, (row1, col1)), shape=(2, 5)),
             sp.coo_matrix((data2, (row2, col2)), shape=(2, 3))]

        n_matrix = nm.NormalizedMatrix(s, r, k, second_order=True)
        rr = (np.asarray(r[0].toarray()[k[0]][:, np.newaxis]) * np.asarray(r[1].toarray()[k[1]])[..., np.newaxis]).reshape(s.shape[0], -1)
        m = np.hstack([n_matrix.ent_table.toarray()] + [n_matrix.att_table[0].toarray()[k[0]],
                                                        n_matrix.att_table[1].toarray()[k[1]], rr])
        assert_almost_equal(np.sort(n_matrix.T * x), np.sort(m.T.dot(x)))

    def test_rmm(self):
        n_matrix = self.n_matrix
        x = np.matrix([[1.0, 2.0, 3.0, 4.0, 5.0]])

        assert_almost_equal(x * n_matrix, x * self.m)

        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        x = np.matrix(np.arange(1.0, 6.0))
        n_matrix = nm.NormalizedMatrix(s, r, k, second_order=True)

        rr = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(r[1][k[1]])[..., np.newaxis]).reshape(s.shape[0], -1)
        sr0 = (np.asarray(s[:, np.newaxis]) * np.asarray(r[0][k[0]])[..., np.newaxis]).reshape(s.shape[0], -1)
        sr1 = (np.asarray(s[:, np.newaxis]) * np.asarray(r[1][k[1]])[..., np.newaxis]).reshape(s.shape[0], -1)

        m = np.hstack([n_matrix.ent_table] + [n_matrix.att_table[0][k[0]], n_matrix.att_table[1][k[1]], sr0, sr1, rr])
        assert_almost_equal(np.sort(x * n_matrix), np.sort(x.dot(m)))
        assert_almost_equal(np.sort(x * (n_matrix + 2)), np.sort(x.dot(m + 2)))
        assert_almost_equal(np.sort(x * (n_matrix * 2)), np.sort(x.dot(m * 2)))

        # rmm with matrix
        x = np.vstack((np.matrix(np.arange(1.0, 6.0)), np.matrix(np.arange(7.0, 12.0))))
        assert_almost_equal(np.sort(x * n_matrix), np.sort(x.dot(m)))
        assert_almost_equal(np.sort(x * (n_matrix + 2)), np.sort(x.dot(m + 2)))
        assert_almost_equal(np.sort(x * (n_matrix * 2)), np.sort(x.dot(m * 2)))

        # # sparse
        # indptr = np.array([0, 2, 3, 6, 6])
        # indices = np.array([0, 2, 2, 0, 1, 4])
        # data1 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        # s = sp.csc_matrix((data1, indices, indptr), shape=(5, 4)).tocoo()
        #
        # row1 = np.array([0, 1, 1])
        # col1 = np.array([0, 4, 1])
        # data1 = np.array([1.0, 2.0, 3.0])
        #
        # row2 = np.array([1, 1, 0])
        # col2 = np.array([2, 1, 1])
        # data2 = np.array([1.1, 2.2, 3.3])
        #
        # k = [np.array([0, 1, 1, 0, 1]), np.array([1, 0, 0, 1, 1])]
        # r = [sp.coo_matrix((data1, (row1, col1)), shape=(2, 5)),
        #      sp.coo_matrix((data2, (row2, col2)), shape=(2, 3))]
        #
        # x = np.matrix(np.arange(1.0, 6.0))
        # n_matrix = nm.NormalizedMatrix(s, r, k, second_order=True)
        # rr = base.data_interaction_rr(r[0].tocsr()[k[0]], r[1].tocsr()[k[1]])
        # sr0 = base.data_interaction_rr(s.tocsr(), r[0].tocsr()[k[0]])
        # sr1 = base.data_interaction_rr(s.tocsr(), r[1].tocsr()[k[1]])
        #
        # m = np.hstack([n_matrix.ent_table.toarray()] + [n_matrix.att_table[0].toarray()[k[0]],
        #                                                 n_matrix.att_table[1].toarray()[k[1]],
        #                                                 sr0.toarray(), sr1.toarray(), rr.toarray()])
        #
        # assert_almost_equal(np.sort(x * n_matrix), np.sort(x.dot(m)))

        # rmm with matrix (sparse)
        x = np.vstack((np.matrix(np.arange(1.0, 6.0)), np.matrix(np.arange(7.0, 12.0))))
        assert_almost_equal(np.sort(x * n_matrix), np.sort(x.dot(m)))
        assert_almost_equal(np.sort(x * (n_matrix + 2)), np.sort(x.dot(m + 2)))
        assert_almost_equal(np.sort(x * (n_matrix * 2)), np.sort(x.dot(m * 2)))


        # with identity
        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        x = np.matrix(np.arange(1.0, 6.0))
        n_matrix = nm.NormalizedMatrix(s, r, k, identity=True)

        m = np.hstack([n_matrix.ent_table] + [np.identity(2)[k[0]], n_matrix.att_table[0][k[0]],
                                              np.identity(2)[k[1]], n_matrix.att_table[1][k[1]]])

        assert_almost_equal(np.sort(x * n_matrix), np.sort(x.dot(m)))
        assert_almost_equal(np.sort(x * (n_matrix + 2)), np.sort(x.dot(m + 2)))
        assert_almost_equal(np.sort(x * (n_matrix * 2)), np.sort(x.dot(m * 2)))

        # with identity and second order
        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1]),
             np.array([1, 1, 0, 1, 0])]

        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]]),
             np.matrix([[5.5, 6.6, 7.7], [8.8, 9.9, 10.10]])]

        x = np.matrix(np.arange(1.0, 6.0))
        n_matrix = nm.NormalizedMatrix(s, r, k, identity=True, second_order=True)
        rr = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(r[1][k[1]])[..., np.newaxis]).reshape(s.shape[0], -1)
        sr0 = (np.asarray(s[:, np.newaxis]) * np.asarray(r[0][k[0]])[..., np.newaxis]).reshape(s.shape[0], -1)
        sr1 = (np.asarray(s[:, np.newaxis]) * np.asarray(r[1][k[1]])[..., np.newaxis]).reshape(s.shape[0], -1)

        m = np.hstack([n_matrix.ent_table] + [np.identity(2)[k[0]], n_matrix.att_table[0][k[0]],
                                              np.identity(2)[k[1]], n_matrix.att_table[1][k[1]], sr0, sr1, rr])

        assert_almost_equal(np.sort(x * n_matrix), np.sort(x.dot(m)))
        assert_almost_equal(np.sort(x * (n_matrix + 2)), np.sort(x.dot(m + 2)))
        assert_almost_equal(np.sort(x * (n_matrix * 2)), np.sort(x.dot(m * 2)))

    def test_rmm_trans(self):
        n_matrix = self.n_matrix
        x = np.matrix([[1.0, 2.0, 3.0, 4.0]])
        assert_equal(x * n_matrix.T, x * self.m.T)

    # def test_cross_prod(self):
    #     n_matrix = self.n_matrix.T * self.n_matrix
    #     assert_almost_equal(n_matrix, self.m.T * self.m)
    #
    #     n_matrix = np.multiply(self.n_matrix.T, self.n_matrix)
    #     assert_almost_equal(n_matrix, self.m.T * self.m)
    #
    #     s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
    #     k = [np.array([0, 1, 1, 0, 1])]
    #
    #     r = [np.matrix([[1.1, 2.2], [3.3, 4.4]])]
    #
    #     n_matrix = nm.NormalizedMatrix(s, r, k, second_order=True)
    #     sr = base.data_interaction_rr(s, r[0][k[0]])
    #     # sr0 = (np.asarray(r[0][k[0]][:, np.newaxis]) * np.asarray(s)[..., np.newaxis]).reshape(s.shape[0], -1)
    #
    #     m = np.hstack([n_matrix.ent_table] + [n_matrix.att_table[0][k[0]], sr])
    #     assert_almost_equal(m.T.dot(m), n_matrix.T * n_matrix)

    # def test_cross_prod_trans(self):
    #     n_matrix = self.n_matrix.T
    #     n_matrix = n_matrix.T * n_matrix
    #     assert_almost_equal(n_matrix, self.m * self.m.T)
    #
    def test_max(self):
        n_matrix = self.n_matrix

        assert_equal(n_matrix.max(), self.m.max())
        assert_equal(n_matrix.max(axis=0), self.m.max(axis=0))

    def test_min(self):
        n_matrix = self.n_matrix

        assert_equal(n_matrix.min(), self.m.min())
        assert_equal(n_matrix.min(axis=0), self.m.min(axis=0))

    def test_mean(self):
        n_matrix = self.n_matrix

        assert_equal(n_matrix.mean(), self.m.mean())
        assert_equal(n_matrix.mean(axis=0), self.m.mean(axis=0))

if __name__ == "__main__":
    run_module_suite()
