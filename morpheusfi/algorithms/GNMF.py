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
from sklearn.base import BaseEstimator

class GaussianNMF(BaseEstimator):
    def __init__(self, iterations=20, components=5):
        self.iterations = iterations
        self.components = components

    def fit(self, X, w_init=None, h_init=None):
        self.w = w_init if w_init is not None else np.mat(np.random.rand(X.shape[0], self.components))
        self.h = h_init if h_init is not None else np.mat(np.random.rand(self.components, X.shape[1]))

        """Factorize non-negative matrix."""
        for _ in range(self.iterations):
            self.h = np.multiply(self.h, (self.w.T * X) / (self.w.T * self.w * self.h))
            self.w = np.multiply(self.w, (X * self.h.T) / (self.w * (self.h * self.h.T)))
            # self.h = np.multiply(self.h, (X.T * self.w) / (self.h * (self.w.T * self.w)))
            # self.w = np.multiply(self.w, (X * self.h) / (self.w * (self.h.T * self.h)))
        return self


