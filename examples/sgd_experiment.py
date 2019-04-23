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

import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score
from morpheusfi.normalized_matrix import NormalizedMatrix
from morpheusfi.torch_matrix import SparseMM
from morpheusfi.torch_matrix import NormalizedMM
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import math
import pandas as pd
import time

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_size, num_classes).double())
        self.reset_parameters(math.sqrt(6 / (input_size + num_classes)))

    def reset_parameters(self, limit):
        self.weight.data.uniform_(-limit, limit)

    def forward(self, x):
        if sparse.issparse(x) or isinstance(x, NormalizedMatrix):
            out = F.sigmoid(NormalizedMM(x)(self.weight))
        else:
            out = F.sigmoid(SparseMM(x)(self.weight))
        return out

class LinearRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearRegression, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_size, num_classes).double())
        self.bias = nn.Parameter(torch.Tensor(num_classes)).double()
        self.reset_parameters(math.sqrt(6 / (input_size + num_classes)))

    def reset_parameters(self, limit):
        self.weight.data.uniform_(-limit, limit)
        self.bias.data.fill_(0)

    def forward(self, x):
        out = torch.add(SparseMM(x)(self.weight), self.bias)
        return out

def CSRtoSparseTensor(m):
    m = m.tocoo()
    i = torch.LongTensor([m.row, m.col])
    v = torch.DoubleTensor(m.data)
    return torch.sparse.DoubleTensor(i, v, torch.Size(m.shape))

def hingeLoss(out, y):
    return torch.mean(torch.clamp(1 - out * y, min=0))

class Experiment(object):
    def __init__(self, args):
        self.args = args
        if 'lr' in args:
            self.learning_rates = [args['lr']]
        else:
            # if args['optimizer'] == 'bgd':
            #     self.learning_rates = [1, 0.5, 0.1, 0.05, 0.01]
            # elif args['optimizer'] == 'adam':
            #     self.learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
            # else:
            #     self.learning_rates = [1, 0.5, 0.1, 0.05, 0.01]
            self.learning_rates = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.00001]

        self.l2_norms = ([args['l2']] if 'l2' in args else [1, 0.5, 0.1, 0.05, 0.01])
        self.logs = pd.DataFrame(columns=[
                             "optimizer",
                             "learning_rate",
                             "l2_norm",
                             "epoch",
                             "iteration",
                             "fit_time",
                             "train_loss",
                             "validation_loss",
                             "validation_accuracy",
                             "error_rate"])
        self.index = 0
        self.parse_data(args['interacted'])

    def save_log(self, lr, l2, epoch, iteration, fit_time, train_loss, validation_loss, validation_acc):
        if lr is None:
            lr = self.lr
        if l2 is None:
            l2 = self.l2
        self.logs.loc[self.index] = [self.args['optimizer'], lr, l2, epoch, iteration,
                                fit_time, train_loss, validation_loss, validation_acc, 1 - validation_acc]
        self.index += 1

    def save_log_csv(self):
        path = self.args['model']  + '/' + self.args['dataset'] + '/' + 'interacted_' + str(self.args['interacted'])
        filename = self.args['optimizer'] + '_' + self.args['system'] + '.csv'
        if not os.path.exists(path):
            os.makedirs(path)

        self.logs.to_csv(path + '/' + filename, encoding='utf-8', index=False)

    def parse_data(self, interacted=False):
        path = './data/Yelp/'


        r1 = sparse.load_npz(path + 'r1.npz')
        r2 = sparse.load_npz(path + 'r2.npz')
        k1_train = np.genfromtxt(path + 'k1_train.csv', dtype=int)
        k1_test = np.genfromtxt(path + 'k1_test.csv', dtype=int)
        k2_train = np.genfromtxt(path + 'k2_train.csv', dtype=int)
        k2_test = np.genfromtxt(path + 'k2_test.csv', dtype=int)

        r1, r2 = r2, r1
        k1_train, k2_train = k2_train, k1_train
        k1_test, k2_test = k2_test, k1_test

        if interacted:
            X_train = NormalizedMatrix(np.matrix([]), [r1.tocoo(), r2.tocoo()], [k1_train, k2_train], second_order=True)
            X_test = NormalizedMatrix(np.matrix([]), [r1.tocoo(), r2.tocoo()], [k1_test, k2_test], second_order=True)
        else:
            X_train = NormalizedMatrix(np.matrix([]), [r1.tocoo(), r2.tocoo()], [k1_train, k2_train])
            X_test = sparse.hstack([r1.tocsr()[k1_test], r2.tocsr()[k2_test]])

        y_train = np.matrix(np.genfromtxt(path + 'y_train.csv', dtype=np.double)).T
        y_test = np.matrix(np.genfromtxt(path + 'y_test.csv', dtype=np.double)).T
        y_train = torch.from_numpy(y_train).double()
        y_test = torch.from_numpy(y_test).double()

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def build_model(self, model, input_dim, output_dim):
        if model == 'lr':
            return LogisticRegression(input_dim, output_dim), torch.nn.BCELoss()

        elif model == 'svm':
            return LinearRegression(input_dim, output_dim), hingeLoss

    def run_model(self, model, lr, l2, optimizer, X_train, y_train, X_test, y_test):
        reg, criterion = self.build_model(model, X_train.shape[1], 1)

        print "system", self.args['system'], ";dataset", self.args['dataset'], "; optimizer", optimizer, "; lr", lr, "; l2", l2

        if optimizer == 'bgd':
            epochs = 2000
            fit_time = 0.0

            for e in range(epochs):
                print "epoch", e
                fit_start = time.time()

                reg.zero_grad()
                y_pred = reg(X_train)
                train_loss = criterion(y_pred, y_train)
                # add l2 norm
                loss = train_loss + l2 * torch.mean(reg.weight ** 2)
                loss.backward()

                # take steps
                with torch.no_grad():
                    for param in reg.parameters():
                        param.data.sub_(lr * param.grad.data)

                fit_time += (time.time() - fit_start)
                los, acc = self.validate(reg, criterion, X_test, y_test)
                self.save_log(lr, l2, e, e, fit_time, train_loss.item(), los, acc)
        elif optimizer == 'lbfgs':
            optimizer = optim.LBFGS(reg.parameters(), max_iter=400, lr=lr, history_size=20)
            self.fit_time = 0.0
            self.counter = 0

            self.train(reg, criterion, optimizer, X_train, y_train, X_test, y_test, l2)

            # save last epoch's information
            print 'iter', self.counter
            los, acc = self.validate(reg, criterion, X_test, y_test)
            self.save_log(None, None, self.counter - 1, self.counter - 1, self.fit_time, self.train_loss, los, acc)
        elif optimizer == 'adam':
            epochs = 10
            optimizer = optim.Adam(reg.parameters(), lr=lr)
            batch_size = 50
            num_batches = X_train.shape[0] // batch_size
            fit_time = 0.0
            tensors = []

            _X = X_train.tocsr()
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                tensors.append(CSRtoSparseTensor(_X[start:end].tocoo()))
            print 'finished processing data'

            for e in range(epochs):
                print "epoch", e

                for k in range(num_batches):
                    start, end = k * batch_size, (k + 1) * batch_size
                    t = time.time()
                    self.train(reg, criterion, optimizer, tensors[k], y_train[start:end], None, None, l2)
                    fit_time += time.time() - t

                    if k % (num_batches / 100) == 0:
                        print 'iter', e * num_batches + k, ';fit time', fit_time
                        los, acc = self.validate(reg, criterion, X_test, y_test)
                        self.save_log(lr, l2, e, e * num_batches + k, fit_time, self.train_loss ,los, acc)

        self.save_log_csv()

    def train(self, model, loss_func, optimizer, x_val, y_val, x_test, y_test, l2):
        self.cur_time = time.time()


        def closure():
            self.fit_time += (time.time() - self.cur_time)
            print 'fit_time is', self.fit_time
            if self.counter > 0:
                print 'iter', self.counter
                los, acc = self.validate(model, loss_func, x_test, y_test)
                self.save_log(None, None, self.counter - 1, self.counter - 1, self.fit_time, self.train_loss, los, acc)
                self.index += 1
            self.counter += 1

            self.cur_time = time.time()

            optimizer.zero_grad()
            x = x_val
            y = Variable(y_val)

            out = model(x)
            train_loss = loss_func(out, y)
            loss = train_loss + l2 * torch.mean(model.weight ** 2)
            loss.backward()

            self.train_loss = train_loss.item()
            return loss

        def closure_wo_validation():
            optimizer.zero_grad()
            x = x_val
            y = Variable(y_val)

            out = model(x)
            train_loss = loss_func(out, y)
            loss = train_loss + l2 * torch.mean(model.weight ** 2)
            loss.backward()

            self.train_loss = train_loss.item()

            return loss

        if self.args['optimizer'] == 'lbfgs':
            optimizer.step(closure)
        else:
            optimizer.step(closure_wo_validation)


    def validate(self, model, loss, x_val, y_val):
        x = x_val
        y = Variable(y_val, requires_grad=False)

        # Forward
        fx = model.forward(x)
        def map_prob_label(x):
            if x > 0.5:
                return 1
            else:
                return 0

        vfunc = np.vectorize(map_prob_label)
        predictions = fx.data.numpy()
        output = loss(fx, y)
        los = output.item()

        acc = accuracy_score(y_val.numpy().astype(int), vfunc(predictions))

        print("validation loss = %f, acc = %f" % (los, acc))
        return los, acc

    def start_experiment(self):
        print 'started the experiment'
        for lr in self.learning_rates:
            for l2 in self.l2_norms:
                self.lr = lr
                self.l2 = l2
                self.run_model(self.args['model'], lr, l2, self.args['optimizer'], self.X_train, self.y_train,
                               self.X_test, self.y_test)

if __name__ == "__main__":
    args = {'dataset': 'movie',
            'model': 'lr',
            'optimizer': 'lbfgs',
            'system': 'X',
            'interacted': True,
            'lr': 1,
            'l2': 0.1}

    experiment = Experiment(args)
    experiment.start_experiment()
