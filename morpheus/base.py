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
import scipy.sparse as sp
from scipy.sparse.linalg import inv
import comp

def data_interaction(m):
    if sp.issparse(m):
        m = m.tocoo()
        n, d = m.shape[0], m.shape[1]
        row_count = np.diff(m.tocsr().indptr)
        nz_count = (np.multiply(row_count + 3, row_count)).sum() / 2
        rows = np.zeros((nz_count,), dtype=np.int32)
        cols = np.zeros((nz_count,), dtype=np.int32)
        data = np.zeros((nz_count,), dtype=float)
        comp.data_interaction_sparse(n, d, m.size, m.row, m.col, m.data, rows, cols, data)

        return sp.coo_matrix((data, (rows, cols)), shape=(n, (d + 3) * d / 2))
    else:
        m = np.asarray(m, order='F')
        res = np.empty((m.shape[0], m.shape[1] + m.shape[1] * (m.shape[1] + 1) / 2), order='F')
        comp.data_interaction(m.shape[0], m.shape[1], m, res)

        return res

def data_interaction_sr(m, n):
    if all([sp.issparse(m)] + map(sp.issparse, n)):
        m = m.tocsc()
        n = [x.tocsc() for x in n]
        sr = []
        for x in n:
            indptr, indices, data = \
                comp.data_interaction_sr_sparse(m.indptr.shape[0], m.indices.shape[0], m.indptr, m.indices, m.data,
                                               x.indptr.shape[0], x.indices.shape[0], x.indptr, x.indices, x.data)
            sr.append(sp.csc_matrix((data, indices, indptr)).tocoo())

    else:
        sr = [(np.asarray(m[:, np.newaxis]) *
               np.asarray(x)[..., np.newaxis]).reshape(m.shape[0], -1) for k, x in n]

    return sr

def data_interaction_rr(m, n):
    if sp.issparse(m) and sp.issparse(n):
        m = m.tocoo()
        n = n.tocoo()

        row_count_m = np.diff(m.tocsr().indptr)
        row_count_n = np.diff(n.tocsr().indptr)
        nz_count = np.multiply(row_count_m, row_count_n).sum()

        rows = np.zeros((nz_count,), dtype=np.int64)
        cols = np.zeros((nz_count,), dtype=np.int64)
        data = np.zeros((nz_count,), dtype=float)

        # comp.data_interaction_rr_sparse(m.shape[0], n.shape[1], m.shape[1], n.size, m.size,
        #                                 n.row.astype(np.int64), n.col.astype(np.int64), n.data,
        #                                 m.row.astype(np.int64), m.col.astype(np.int64), m.data,
        #                                 rows, cols, data)
        comp.data_interaction_rr_sparse(m.shape[0], m.shape[1], n.shape[1], m.size, n.size,
                                        m.row.astype(np.int64), m.col.astype(np.int64), m.data,
                                        n.row.astype(np.int64), n.col.astype(np.int64), n.data,
                                        rows, cols, data)
        return sp.coo_matrix((data, (rows, cols)), shape=(m.shape[0], m.shape[1] * n.shape[1]))
    else:
        return (np.asarray(m[:, np.newaxis]) * np.asarray(n)[..., np.newaxis]).reshape(m.shape[0], -1)
    # if sp.issparse(m) and sp.issparse(n):
    #     m = m.tocsc()
    #     n = n.tocsc()
    #     indptr, indices, data = \
    #         comp.data_interaction_sr_sparse(m.indptr.shape[0], m.indices.shape[0], m.indptr, m.indices, m.data,
    #                                         n.indptr.shape[0], n.indices.shape[0], n.indptr, n.indices, n.data)
    #     return sp.csc_matrix((data, indices, indptr)).tocoo()
    #
    # else:
    #     return (np.asarray(m[:, np.newaxis]) * np.asarray(n)[..., np.newaxis]).reshape(m.shape[0], -1)
