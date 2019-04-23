// Copyright 2019 Side Li, Lingjiao Chen and Arun Kumar
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Python.h>
#include <vector>
#include <unordered_map>
#include <map>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <tuple>
#include <functional>
#include <math.h>
#include <numpy/arrayobject.h>
#include <cstdio>
#include <ctime>
extern "C" {
    void initcomp(void);
}

template <class I, class T, class K>
static void add_new(const I ns,
                    const I nk,
                    const I dw,
                    const std::vector<K*> k,
                    const std::vector<T*> v,
                    const std::vector<K> vd,
                          T res[])
{
    for (int j = 0; j < nk; j++) {
        for (int c = 0; c < dw; c++) {
            long o1 = c * ns;
            long o2 = c * vd[j];
            for (int i = 0; i < ns; i++) {
                res[o1++] += v[j][k[j][i] + o2];
            }
        }
    }
//  Another implementation that exploits other memory layout
//    long o = 0;
//    for (int i = 0; i < ns; i++) {
//        for (int j = 0; j < dw; j++) {
//            for (int m = 0; m < nk; m++) {
//                res[o] += v[m][k[m][i] * vd[m] + j];
//            }
//            o++;
//        }
//    }
}

template <class I, class T, class K>
static void add_new_identity(const I ns,
                    const I nk,
                    const I dw,
                    const std::vector<K*> k,
                    const std::vector<T*> v,
                    const std::vector<K> vd,
                          T res[])
{
    for (int j = 0; j < nk * 2; j++) {
        int m = j / 2;
        for (int c = 0; c < dw; c++) {
            long o1 = c * ns;
            long o2 = c * vd[j];
            for (int i = 0; i < ns; i++) {
                res[o1++] += v[j][k[m][i] + o2];
            }
        }
    }
//  Another implementation that exploits other memory layout
//    long o = 0;
//    for (int i = 0; i < ns; i++) {
//        for (int j = 0; j < dw; j++) {
//            for (int m = 0; m < nk; m++) {
//                res[o] += v[m][k[m][i] * vd[m] + j];
//            }
//            o++;
//        }
//    }
}

template <class I, class T, class K>
static void expand_add(const I ns,
                    const I nk,
                    const std::vector<K*> k,
                    const std::vector<T*> r,
                    const std::vector<K> nr,
                          T res[])
{
    long o = 0;
    for (int i = 0; i < ns; i++) {
        for (int j = 0; j < ns; j++) {
            for (int m = 0; m < nk; m++) {
                res[o++] += r[m][nr[m] * k[m][i] + k[m][j]];
            }
        }
    }
}

template <class I, class T, class K>
static void group_left(const I ns,
                    const I ds,
                    const T* s,
                    const K* k,
                          T res[])
{
    long o1 = 0;
    for (int i = 0; i < ns; i++) {
        long o2 = k[i] * ds;
        for (int j = 0; j < ds; j++) {
            res[o2++] += s[o1++];
        }
    }
}

template <class I, class T, class K>
static void group(const I ns,
                  const I nk,
                  const I nw,
                  const std::vector<K*> k,
                  const std::vector<I> nr,
                  const T* w,
                        std::vector<T*> res)
{
    for (int j = 0; j < nk; j++) {
        long o1 = 0;
        long o2 = 0;
        for (int r = 0; r < nw; r++) {
            for (int i = 0; i < ns; i++) {
                res[j][o1 + k[j][i]] += w[o2++];
            }
            o1 += nr[j];
        }
    }
}

template <class I, class T>
static void multiply(const I nr,
                     const I dr,
                     const T* r,
                     const T* v,
                     T* res)
{
    int o = 0;
    for (int i = 0; i < nr; i++) {
        T scalar = sqrt(v[i]);
        for (int j = 0; j < dr; j++) {
            res[o] = r[o] * scalar;
            o++;
        }
    }
}

template <class I, class T>
static void multiply_sparse(const I n,
                     const I* rows,
                     const T* data,
                     const T* v,
                     T* res)
{
    for (int i = 0; i < n; i++) {
        res[i] = data[i] * v[rows[i]];
    }
}

template <class I, class T>
static void left_dot_sparse(const I n,
                     const I offset,
                     const T* other,
                     const I* row,
                     const I* col,
                     const T* data,
                     T* res)
{
    for (int i = 0; i < n; i++) {
        res[col[i] + offset] += data[i] * other[row[i]];
    }
}

template <class I, class T>
static void data_interaction(const I n,
                     const I d,
                     const T* data,
                     T* res)
{
    long c = 0;
    // copy 1st order
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            res[c++] = data[i * n + j];
        }
    }

    // create 2nd order interaction
    for (int i = 0; i < d; i++) {
        for (int j = i; j < d; j++) {
            for (int k = 0; k < n; k++) {
                res[c++] = data[i * n + k] * data[j * n + k];
            }
        }
    }
}

template <class I, class T>
static void data_interaction_sparse(const I n,
                     const I d,
                     const I nz,
                     const I* rows,
                     const I* cols,
                     const T* data,
                     I* newRow,
                     I* newCol,
                     T* newData)
{
    // create rows of maps
    std::vector<std::vector<std::pair<I, T> > > rows1(n);

    // initialize r1 map and copy 1st order
    for (I i = 0; i < nz; i++) {
        rows1[rows[i]].push_back(std::pair<I, T>(cols[i], data[i]));

        // 1st order
        newRow[i] = rows[i];
        newCol[i] = cols[i];
        newData[i] = data[i];
    }

    int ind = nz;

    // copy 1st order
    for (I i = 0; i < n; i++) {
        std::vector<std::pair<I, T> > row = rows1[i];
        for (I j = 0; j < row.size(); j++) {
            std::pair<I, T> ele1 = row[j];
            int base = d + (d * 2 - ele1.first + 1) * ele1.first / 2 - ele1.first;
            for (I k = j; k < row.size(); k++) {
                std::pair<I, T> ele2 = row[k];

                newRow[ind] = i;
                newData[ind] = ele1.second * ele2.second;
                newCol[ind] = base + ele2.first;
                ind++;
            }
        }
    }
}

template <class I, class T>
static void data_interaction_rr_sparse(const I n,
                     const I d1,
                     const I d2,
                     const I nz1,
                     const I nz2,
                     const I* rows1,
                     const I* cols1,
                     const T* data1,
                     const I* rows2,
                     const I* cols2,
                     const T* data2,
                     I* newRow,
                     I* newCol,
                     T* newData)
{
    // create rows of maps
    std::vector<std::vector<std::pair<I, T> > > rowsMap1(n);
    std::vector<std::vector<std::pair<I, T> > > rowsMap2(n);

    // initialize r1 map
    for (I i = 0; i < nz1; i++) {
        rowsMap1[rows1[i]].push_back(std::pair<I, T>(cols1[i], data1[i]));
    }
    for (I i = 0; i < nz2; i++) {
        rowsMap2[rows2[i]].push_back(std::pair<I, T>(cols2[i], data2[i]));
    }

    I ind = 0;

    // 2nd order
    for (I i = 0; i < n; i++) {
        std::vector<std::pair<I, T> > row1 = rowsMap1[i];
        std::vector<std::pair<I, T> > row2 = rowsMap2[i];

        for (I j = 0; j < row1.size(); j++) {
            std::pair<I, T> ele1 = row1[j];
            I base = ele1.first * d2;
            for (I k = 0; k < row2.size(); k++) {
                std::pair<I, T> ele2 = row2[k];

                newRow[ind] = i;
                newData[ind] = ele1.second * ele2.second;
                newCol[ind] = base + ele2.first;
                ind++;
            }
        }
    }
}

template <class I, class T>
static void data_interaction_sr_sparse(const I n1,
                     const I d1,
                     const I n2,
                     const I d2,
                     const I* indptr1,
                     const I* indices1,
                     const T* data1,
                     const I* indptr2,
                     const I* indices2,
                     const T* data2,
                     std::vector<I> & newIndptr,
                     std::vector<I> & newIndices,
                     std::vector<T> & newData)
{
    // create rows of maps
    I pre = indptr1[0];
    std::vector< std::unordered_map<I, T> > cols1;
    std::vector< std::unordered_map<I, T> > cols2;
    for (I i = 1; i < n1; i++) {
        std::unordered_map<I, T> col;
        for (I j = pre; j < indptr1[i]; j++) {
            col.insert(std::pair<I, T>(indices1[j], data1[j]));
        }
        cols1.push_back(col);
        pre = indptr1[i];
    }

    pre = indptr2[0];
    for (I i = 1; i < n2; i++) {
        std::unordered_map<I, T> col;
        for (I j = pre; j < indptr2[i]; j++) {
            col.insert(std::pair<I, T>(indices2[j], data2[j]));
        }
        cols2.push_back(col);
        pre = indptr2[i];
    }

    I numCols1 = cols1.size();
    I numCols2 = cols2.size();
    newIndptr.push_back(0);
    for (I i = 0; i < numCols1; i++) {
        for (I j = 0; j < numCols2; j++) {
            // find intersection
            std::unordered_map<I, T> v_intersection;
            for (auto const& x : cols1[i]) {
                I key = x.first;
                T value = x.second;
                if (cols2[j].find(key) != cols2[j].end()) {
                    v_intersection.insert(std::pair<I, T>(key, value * cols2[j][key]));
                }
            }
            // add to indptr, indices and data
            for (auto const& x : v_intersection) {
                newIndices.push_back(x.first);
                newData.push_back(x.second);
            }

            newIndptr.push_back(newIndptr.back() + v_intersection.size());
        }
    }
}

template<class I, class T, class K>
static void hadamard_rowsum(const I ns,
                            const I dr,
                            const I dx,
                            const I sr,
                            const T* k1,
                            const T* k2,
                            const K* r1,
                            const K* r2,
                            K* res)
{
    if (sr) {
        for (I i = 0; i < ns; i++) {
            T base1 = k1[i] * dr * dx;
            T base2 = i * dr;

            for (I m = 0; m < dx; m++) {
                I iBase = i + m * ns;

                for (I j = 0; j < dr; j++) {
                    res[iBase] += (r1[j + base1] * r2[j + base2]);
                }

                base1 += dr;
            }
        }
    } else {
        for (I i = 0; i < ns; i++) {
            T base1 = k1[i] * dr * dx;
            T base2 = k2[i] * dr;

            for (I m = 0; m < dx; m++) {
                I iBase = i + m * ns;

                for (I j = 0; j < dr; j++) {
                    res[iBase] += (r1[j + base1] * r2[j + base2]);
                }

                base1 += dr;
            }
        }
    }

}

template<class I, class T, class K>
static void hadamard_rowsum_sparse(const I ns,
                            const I nz,
                            const I nr1,
                            const I nr2,
                            const I dr,
                            const I dx,
                            const I sr,
                            const T* k1,
                            const T* k2,
                            const K* r1,
                            const I* row,
                            const I* col,
                            const K* data,
                            K* res)
{
    if (sr) {
        for (I i = 0; i < nz; i++) {
            I base = col[i] * nr1;
            K d = data[i];

            for (I m = 0; m < dx; m++) {
                res[row[i] + ns * m] += r1[base + k1[row[i]]] * d;
                base += nr1 * dr;
            }
        }
    } else {
        std::vector< std::vector<I> > maskedCols;
        for (I i = 0; i < nr2; i++) {
            std::vector<I> v;
            maskedCols.push_back(v);
        }

        for (I i = 0; i < ns; i++) {
            maskedCols[k2[i]].push_back(i);
        }

        for (I i = 0; i < nz; i++) {
            I base = col[i] * nr1;
            K d = data[i];

            for (I m = 0; m < dx; m++) {
                I iBase = ns * m;
                for (auto& it : maskedCols[row[i]]) {
                    res[it + iBase] += r1[base + k1[it]] * d;
                }
                base += nr1 * dr;
            }
        }
    }
}

template<class I, class T, class K>
static void data_interaction_group(const I ns,
                            const I dr1,
                            const I nw,
                            const T* k1,
                            const T* k2,
                            const K* r1,
                            const K* w,
                            K* res)
{
    for (I i = 0; i < ns; i++) {
        T base1 = k1[i] * dr1;
        T base2 = k2[i] * dr1 * nw;
        I baseW = i;

        for (I m = 0; m < nw; m++) {
            for (I j = 0; j < dr1; j++) {
                res[base2 + j] += r1[base1 + j] * w[baseW];
            }
            base2 += dr1;
            baseW += ns;
        }
    }
}

template<class I, class T, class K>
static void data_interaction_group_sparse(const I nz1,
                     const I n1,
                     const I d1,
                     const I nz2,
                     const I n2,
                     const I d2,
                     const I ns,
                     const I nw,
                     const I* row1,
                     const I* col1,
                     const K* data1,
                     const I* row2,
                     const I* col2,
                     const K* data2,
                     const T* k1,
                     const T* k2,
                     const K* w,
                     K* res)
{
    std::vector< std::unordered_map<I, K> > rows1;
    std::vector< std::unordered_map<I, K> > rowsTmp;
    std::vector< std::unordered_map<I, K> > rows2;

    // create transformed r1
    if (ns == n1) {
        for (I i = 0; i < n2; i++) {
            std::unordered_map<I, K> rowB;
            rows2.push_back(rowB);
        }

        // initialize r2 map
        for (I i = 0; i < nz2; i++) {
            rows2[row2[i]].insert(std::pair<I, K>(col2[i], data2[i]));
        }

        K* r1 = (K*) calloc(n2 * d1 * nw, sizeof(K));

        // initialize r1 map
        for (I i = 0; i < nz1; i++) {
            I row = row1[i];
            I col = col1[i];
            K data = data1[i];
            I base = k2[row] * d1 * nw + col;

            for (I m = 0; m < nw; m++) {
                r1[base + m * d1] += data * w[row + m * ns];
            }
        }

        for (I i = 0; i < n2; i++) {
            for (I j = 0; j < d1 * nw; j++) {
                I base = j * d2;
                K data = r1[i * d1 * nw + j];

                for (auto const& x: rows2[i]) {
                    res[base + x.first] += x.second * data;
                }
            }
        }
        delete [] r1;
    } else {
        for (I i = 0; i < n2; i++) {
            std::unordered_map<I, K> rowB;
            rows2.push_back(rowB);
        }

        // initialize r2 map
        for (I i = 0; i < nz2; i++) {
            rows2[row2[i]].insert(std::pair<I, K>(col2[i], data2[i]));
        }

        K* r1 = (K*) calloc(n2 * d1 * nw, sizeof(K));

        // initialize r1 map
        for (I i = 0; i < nz1; i++) {
            I row = row1[i];
            I col = col1[i];
            K data = data1[i];
            I base = k2[row] * d1 * nw + col;

            for (I m = 0; m < nw; m++) {
                r1[base + m * d1] += data * w[row + m * ns];
            }
        }

        for (I i = 0; i < n2; i++) {
            for (I j = 0; j < d1 * nw; j++) {
                I base = j * d2;
                K data = r1[i * d1 * nw + j];

                for (auto const& x: rows2[i]) {
                    res[base + x.first] += x.second * data;
                }
            }
        }
    }
}


static PyObject * add_new(PyObject *self, PyObject* args)
{
    int identity;
    int ns;
    int nk;
    int dw;
    PyObject* k;
    PyObject* vd;
    PyObject* v;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiiiOOOO", &identity, &ns, &nk, &dw, &k, &v, &vd, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    std::vector<long*> k_list;
    std::vector<double*> v_list;
    std::vector<long> vd_list;
    for(int i = 0; i < nk; i++) {
      k_list.push_back((long*) PyArray_DATA(PyList_GET_ITEM(k, i)));
      v_list.push_back((double*) PyArray_DATA(PyList_GET_ITEM(v, i)));
      vd_list.push_back(PyInt_AsLong(PyList_GET_ITEM(vd, i)));
    }

    if (identity) {
        for(int i = nk; i < nk * 2; i++) {
          v_list.push_back((double*) PyArray_DATA(PyList_GET_ITEM(v, i)));
          vd_list.push_back(PyInt_AsLong(PyList_GET_ITEM(vd, i)));
        }
        add_new_identity<int, double, long>(ns, nk, dw, k_list, v_list, vd_list, (double*)PyArray_DATA(res));
    } else {
        add_new<int, double, long>(ns, nk, dw, k_list, v_list, vd_list, (double*)PyArray_DATA(res));
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * group_left(PyObject *self, PyObject* args)
{
    int ns;
    int ds;
    PyObject* k;
    PyObject* s;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiOOO", &ns, &ds, &s, &k, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    group_left<int, double, long>(ns, ds, (double*) PyArray_DATA(s), (long*) PyArray_DATA(k), (double*)PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * expand_add(PyObject *self, PyObject* args)
{
    int ns;
    int nk;
    PyObject* k;
    PyObject* r;
    PyObject* nr;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiOOOO", &ns, &nk, &k, &r, &nr, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    std::vector<long*> k_list;
    std::vector<double*> r_list;
    std::vector<long> nr_list;
    for(int i = 0; i < nk; i++) {
          k_list.push_back((long*) PyArray_DATA(PyList_GET_ITEM(k, i)));
          r_list.push_back((double*) PyArray_DATA(PyList_GET_ITEM(r, i)));
          nr_list.push_back(PyInt_AsLong(PyList_GET_ITEM(nr, i)));
    }

    expand_add<int, double, long>(ns, nk, k_list, r_list, nr_list, (double*)PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * group(PyObject *self, PyObject* args)
{
    int ns;
    int nk;
    int nw;
    PyObject* k;
    PyObject* w;
    PyObject* res;
    PyObject* nr;

    if (!PyArg_ParseTuple(args, "iiiOOOO", &ns, &nk, &nw, &k, &nr, &w, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    std::vector<long*> k_list;
    std::vector<double*> res_list;
    std::vector<int> nr_list;
    for(int i = 0; i < nk; i++) {
          k_list.push_back((long*) PyArray_DATA(PyList_GET_ITEM(k, i)));
          res_list.push_back((double*) PyArray_DATA(PyList_GET_ITEM(res, i)));
          nr_list.push_back((long) PyInt_AsLong(PyList_GET_ITEM(nr, i)));
    }

    group<int, double, long>(ns, nk, nw, k_list, nr_list, (double*) PyArray_DATA(w), res_list);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * multiply(PyObject *self, PyObject* args)
{
    int nr;
    int dr;
    PyObject* r;
    PyObject* v;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiOOO", &nr, &dr, &r, &v, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    multiply<int, double>(nr, dr, (double*) PyArray_DATA(r), (double *) PyArray_DATA(v), (double *) PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * multiply_sparse(PyObject *self, PyObject* args)
{
    int n;
    PyObject* rows;
    PyObject* data;
    PyObject* v;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iOOOO", &n, &rows, &data, &v, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    multiply_sparse<int, double>(n,
                          (int*) PyArray_DATA(rows), (double*) PyArray_DATA(data),
                          (double *) PyArray_DATA(v), (double *) PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * left_dot_sparse(PyObject *self, PyObject* args)
{
    int n;
    int offset;
    PyObject* other;
    PyObject* row;
    PyObject* col;
    PyObject* data;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iOOOOOi", &n, &other, &row, &col, &data, &res, & offset)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    left_dot_sparse<int, double>(n, offset,
                            (double*) PyArray_DATA(other), (int*) PyArray_DATA(row),
                            (int*) PyArray_DATA(col), (double*) PyArray_DATA(data),
                            (double*) PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * data_interaction(PyObject *self, PyObject* args)
{
    int n;
    int d;
    PyObject* data;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiOO", &n, &d, &data, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    data_interaction<int, double>(n, d, (double *) PyArray_DATA(data), (double *) PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

template<typename T>
static PyArrayObject* vector_to_nparray(const std::vector<T>& vec, int type_num){
   size_t nRows = vec.size();
   npy_intp dims[1] = {nRows};
//   PyArrayObject* vec_array = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, type_num, (void*) vec.data());
//   return vec_array;

    PyArrayObject* vec_array = (PyArrayObject *) PyArray_SimpleNew(1, dims, type_num);
    T *vec_array_pointer = (T*) PyArray_DATA(vec_array);

    copy(vec.begin(),vec.end(),vec_array_pointer);
    return vec_array;
}

static PyObject * data_interaction_sparse(PyObject *self, PyObject* args)
{
    int n;
    int d;
    int nz;
    PyObject* rows;
    PyObject* cols;
    PyObject* data;
    PyObject* newRow;
    PyObject* newCol;
    PyObject* newData;

    if (!PyArg_ParseTuple(args, "iiiOOOOOO", &n, &d, &nz, &rows, &cols, &data, &newRow, &newCol, &newData)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    data_interaction_sparse<int, double>(n, d, nz, (int*) PyArray_DATA(rows), (int*) PyArray_DATA(cols),
                                   (double*) PyArray_DATA(data), (int*) PyArray_DATA(newRow),
                                   (int*) PyArray_DATA(newCol), (double*) PyArray_DATA(newData));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * data_interaction_rr_sparse(PyObject *self, PyObject* args)
{
    long n;
    long d1;
    long d2;
    long nz1;
    long nz2;
    PyObject* rows1;
    PyObject* cols1;
    PyObject* data1;
    PyObject* rows2;
    PyObject* cols2;
    PyObject* data2;
    PyObject* newRow;
    PyObject* newCol;
    PyObject* newData;

    if (!PyArg_ParseTuple(args, "lllllOOOOOOOOO", &n, &d1, &d2, &nz1, &nz2,
                          &rows1, &cols1, &data1,
                          &rows2, &cols2, &data2,
                          &newRow, &newCol, &newData)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    data_interaction_rr_sparse<long, double>(n, d1, d2, nz1, nz2, (long*) PyArray_DATA(rows1),
                                    (long*) PyArray_DATA(cols1), (double*) PyArray_DATA(data1),
                                    (long*) PyArray_DATA(rows2), (long*) PyArray_DATA(cols2),
                                   (double*) PyArray_DATA(data2), (long*) PyArray_DATA(newRow),
                                   (long*) PyArray_DATA(newCol), (double*) PyArray_DATA(newData));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * data_interaction_sr_sparse(PyObject *self, PyObject* args)
{
    int n1;
    int d1;
    int n2;
    int d2;
    PyObject* indptr1;
    PyObject* indices1;
    PyObject* data1;
    PyObject* indptr2;
    PyObject* indices2;
    PyObject* data2;

    if (!PyArg_ParseTuple(args, "iiOOOiiOOO", &n1, &d1, &indptr1, &indices1, &data1, &n2, &d2, &indptr2, &indices2, &data2)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    std::vector<int> newIndptr;
    std::vector<int> newIndices;
    std::vector<double> newData;
    data_interaction_sr_sparse<int, double>(n1, d1, n2, d2,
                                  (int *) PyArray_DATA(indptr1),
                                  (int *) PyArray_DATA(indices1),
                                  (double *) PyArray_DATA(data1),
                                  (int *) PyArray_DATA(indptr2),
                                  (int *) PyArray_DATA(indices2),
                                  (double *) PyArray_DATA(data2),
                                  newIndptr,
                                  newIndices,
                                  newData);

    PyObject *rslt = PyTuple_New(3);

    PyTuple_SET_ITEM(rslt, 0, (PyObject*) vector_to_nparray<int>(newIndptr, NPY_INT));
    PyTuple_SET_ITEM(rslt, 1, (PyObject*) vector_to_nparray<int>(newIndices, NPY_INT));
    PyTuple_SET_ITEM(rslt, 2, (PyObject*) vector_to_nparray<double>(newData, NPY_FLOAT64));
    return rslt;
}

static PyObject * hadamard_rowsum(PyObject *self, PyObject* args)
{
    int ns;
    int dr;
    int dx;
    int sr;
    PyObject* k1;
    PyObject* k2;
    PyObject* r1;
    PyObject* r2;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiiiOOOOO", &ns, &dr, &dx, &sr, &k1, &k2, &r1, &r2, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    hadamard_rowsum<int, long, double>(ns, dr, dx, sr, (long*) PyArray_DATA(k1), (long*) PyArray_DATA(k2),
                                       (double*) PyArray_DATA(r1), (double*) PyArray_DATA(r2),
                                       (double*) PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * hadamard_rowsum_sparse(PyObject *self, PyObject* args)
{
    int ns;
    int nz;
    int nr1;
    int nr2;
    int dr;
    int dx;
    int sr;
    PyObject* k1;
    PyObject* k2;
    PyObject* r1;
    PyObject* row;
    PyObject* col;
    PyObject* data;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiiiiiiOOOOOOO", &ns, &nz, &nr1, &nr2, &dr, &dx, &sr, &k1, &k2, &r1, &row, &col, &data, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    hadamard_rowsum_sparse<int, long, double>(ns, nz, nr1, nr2, dr, dx, sr, (long*) PyArray_DATA(k1), (long*) PyArray_DATA(k2),
                                       (double*) PyArray_DATA(r1),
                                       (int*) PyArray_DATA(row), (int*) PyArray_DATA(col),
                                       (double*) PyArray_DATA(data), (double*) PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * data_interaction_group(PyObject *self, PyObject* args)
{
    int ns;
    int dr;
    int nw;
    PyObject* k1;
    PyObject* k2;
    PyObject* r1;
    PyObject* w;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiiOOOOO", &ns, &dr, &nw, &k1, &k2, &r1, &w, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    data_interaction_group<int, long, double>(ns, dr, nw, (long*) PyArray_DATA(k1), (long*) PyArray_DATA(k2),
                                       (double*)PyArray_DATA(r1), (double*) PyArray_DATA(w),
                                       (double*) PyArray_DATA(res));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * data_interaction_group_sparse(PyObject *self, PyObject* args)
{
    int ns;
    int nz1;
    int n1;
    int d1;
    int nz2;
    int n2;
    int d2;
    int nw;
    PyObject* row1;
    PyObject* col1;
    PyObject* data1;
    PyObject* row2;
    PyObject* col2;
    PyObject* data2;
    PyObject* k1;
    PyObject* k2;
    PyObject* w;
    PyObject* res;

    if (!PyArg_ParseTuple(args, "iiiiiiiiOOOOOOOOOO", &ns, &nz1, &n1, &d1, &nz2, &n2, &d2, &nw,
                                                    &row1, &col1, &data1, &row2, &col2, &data2,
                                                    &k1, &k2, &w, &res)){
        PyErr_SetString(PyExc_ValueError,"Error while parsing the trajectory coordinates in get_spinangle_traj");
        return NULL;
    }

    data_interaction_group_sparse<int, long, double>(nz1, n1, d1, nz2, n2, d2, ns, nw,
                                       (int*) PyArray_DATA(row1), (int*) PyArray_DATA(col1), (double*) PyArray_DATA(data1),
                                       (int*) PyArray_DATA(row2), (int*) PyArray_DATA(col2), (double*) PyArray_DATA(data2),
                                       (long*) PyArray_DATA(k1), (long*) PyArray_DATA(k2),
                                       (double*) PyArray_DATA(w), (double*) PyArray_DATA(res));


    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef comp_methods[] = {
	{"add_new", add_new,    METH_VARARGS,
	 "add v list to res"},
	{"expand_add", expand_add,    METH_VARARGS,
	 "expand nr*ns matrix to ns*ns"},
	{"group_left", group_left, METH_VARARGS,
	 "group ns*ds matrix to nr*ds"},
	{"group", group, METH_VARARGS,
	 "group in rmm"},
	{"multiply", multiply, METH_VARARGS,
	 "multiply dense matrix with scalar vector"},
	{"multiply_sparse", multiply_sparse, METH_VARARGS,
	 "multiply coo sparse matrix with scalar vector"},
	{"data_interaction", data_interaction, METH_VARARGS,
	 "generate 2nd order feature interactions"},
	{"data_interaction_sparse", data_interaction_sparse, METH_VARARGS,
	 "generate 2nd order feature interactions for one coo matrix"},
	{"data_interaction_rr_sparse", data_interaction_rr_sparse, METH_VARARGS,
	 "generate 2nd order feature interactions for two coo matrices"},
	{"data_interaction_sr_sparse", data_interaction_sr_sparse, METH_VARARGS,
	 "generate 2nd order feature interactions between s and r for csc matrix"},
	{"left_dot_sparse", left_dot_sparse, METH_VARARGS,
	 "vector times sparse matrix in rmm"},
	{"hadamard_rowsum", hadamard_rowsum, METH_VARARGS,
	 "perform hadmard and rowsum in one place"},
	{"hadamard_rowsum_sparse", hadamard_rowsum_sparse, METH_VARARGS,
	 "perform hadmard and rowsum in one place for coo matrix"},
	{"data_interaction_group", data_interaction_group, METH_VARARGS,
	 "first data interaction and then group by k1(used in rmm)"},
	{"data_interaction_group_sparse", data_interaction_group_sparse, METH_VARARGS,
	 "first data interaction and then group by k1(used in rmm) for sparse coo matrices"},
	{NULL,		NULL}		/* sentinel */
};

extern void initcomp(void)
{
	PyImport_AddModule("comp");
	Py_InitModule("comp", comp_methods);
	import_array();
}

int main(int argc, char **argv)
{
	Py_SetProgramName(argv[0]);

	Py_Initialize();

	initcomp();

	Py_Exit(0);
}