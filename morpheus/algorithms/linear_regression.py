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
from sklearn.base import BaseEstimator, RegressorMixin

class NormalizedLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, iterations=20, gamma=0.000001):
        self.gamma = gamma
        self.iterations = iterations

    def fit(self, X, y, w_init=None):
        self.w = w_init if w_init is not None else np.matrix(np.random.randn(X.shape[1], 1))
        # import time
        # time_cost = [0] * 5
        for _ in range(self.iterations):
            self.w -= self.gamma * (X.T * (X * self.w - y))
            # start = time.time()
            # tmp0 = X * self.w
            # print 'X * self.w'
            # time_cost[0] += time.time() - start
            # print time_cost[0]
            # print "tmp0 shape", tmp0.shape
            #
            # start = time.time()
            # tmp1 = tmp0 - y
            # print 'tmp0 - y'
            # time_cost[1] += time.time() - start
            # print "tmp1 shape", tmp1.shape
            #
            # start = time.time()
            # tmp2 = X.T * tmp1
            # print 'X.T * tmp1'
            # time_cost[2] += time.time() - start
            #
            # print "tmp2 shape", tmp2.shape
            # start = time.time()
            # tmp3 = self.gamma * tmp2
            # print 'self.gamma * tmp2'
            # time_cost[3] += time.time() - start
            #
            # start = time.time()
            # # print self.w.shape, tmp3.shape
            # print "tmp3 shape", tmp3.shape
            # print "w shape", self.w.shape
            # self.w -= tmp3
            # print 'self.w -= tmp3'
            # time_cost[4] += time.time() - start

        # print "linear regression", time_cost
        return self

    def predict(self, X):
        try:
            getattr(self, "w")
        except AttributeError:
            raise RuntimeError("You must train the regressor before predicting data!")

        return X * self.w


