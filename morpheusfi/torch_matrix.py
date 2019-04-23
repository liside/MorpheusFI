# Copyright 2019 Side Li, Lingjiao and Arun Kumar
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input

class NormalizedMM(torch.autograd.Function):
    def __init__(self, sparse):
        super(NormalizedMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        res = self.sparse.dot(dense.data.numpy())
        return torch.DoubleTensor(res)

    def backward(self, grad_output):
        res = self.sparse.T.dot(grad_output.data.numpy())
        grad_input = torch.DoubleTensor(res)
        return grad_input