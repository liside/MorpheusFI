# MorpheusFI

MopheusFI is a Python implementation of factorized learning described in paper: [Enabling and Optimizing Non-linear Feature Interactions in Factorized Linear Algebra]()

## Prerequisite
- Python 2.7
- NumPy 1.13
- SciPy 1.0.0
- SciKit Learn
- Python developer kit (C++ rewrite is used)
- PyTorch 1.0

## Installation
- To install, simply run `python setup.py install`.
- To build C++ module, run `python setup.py build_ext --inplace`.

## Usage
Simply import `NormalizedMatrix` like a NumPy matrix. Then apply linear algebra operations on it.
E.g

```
s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
k = [np.array([0, 1, 1, 0, 1])]
r = [np.matrix([[1.1, 2.2], [3.3, 4.4]])]
x = np.matrix([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0], [13.0], [14.0]])

# materialized matrix with feature interaction
m = np.matrix([[ 1.  ,  2.  ,  1.  ,  2.  ,  4.  ,  1.1 ,  2.2 ,  1.21,  2.42,  4.84,  1.1 ,  2.2 ,  2.2 ,  4.4 ],
               [ 4.  ,  3.  , 16.  , 12.  ,  9.  ,  3.3 ,  4.4 , 10.89, 14.52,  19.36, 13.2 , 17.6 ,  9.9 , 13.2 ],
               [ 5.  ,  6.  , 25.  , 30.  , 36.  ,  3.3 ,  4.4 , 10.89, 14.52,  19.36, 16.5 , 22.  , 19.8 , 26.4 ],
               [ 8.  ,  7.  , 64.  , 56.  , 49.  ,  1.1 ,  2.2 ,  1.21,  2.42,  4.84,  8.8 , 17.6 ,  7.7 , 15.4 ],
               [ 9.  ,  1.  , 81.  ,  9.  ,  1.  ,  3.3 ,  4.4 , 10.89, 14.52,  19.36, 29.7 , 39.6 ,  3.3 ,  4.4 ]])

# normalized matrix
n_matrix = nm.NormalizedMatrix(s, r, k, second_order=true)

# result:
n_matrix.dot(x):
array([[ 266.56],
       [1282.9 ],
       [1926.5 ],
       [1408.56],
       [1663.4 ]])

m_matrix.dot(x):
array([[ 266.56],
       [1282.9 ],
       [1926.5 ],
       [1408.56],
       [1663.4 ]])

```    
## Operators supported
add, substract, divide, multiply, dot product, cross product, inverse

## Note
This library is implemented as a high-level wrapper. It might conflict with existing machine learning libraries if the libraries skip high-level implementation to optimize linear algebra in C kernel.




