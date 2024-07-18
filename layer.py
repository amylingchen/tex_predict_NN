'''
# -*- coding: utf-8 -*-
    @Author   : LingLing
    @Time     : 2024/7/3 22:50
    @File     : layer.py
    @Project  : assignment3
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

class Layer:
    def forward(self, X):
        raise NotImplementedError

    def backward(self, dY):
        raise NotImplementedError


np.random.seed(666)


# class Linear1(Layer):
#     def __init__(self, input_size, output_size, learning_rate=0.01,m_l1= 0):
#         '''
#         Linear layer
#         :param
#             input_size: input size
#             output_size: output size
#             learning_rate: learning rate
#             m_l1: l1 regularization
#
#         '''
#         self.output = None
#         self.input = None
#         # he initialization
#         self.weights = np.random.randn(input_size, output_size) *np.sqrt(2./input_size)
#         self.bias = np.zeros((1, output_size))
#         self.learning_rate = learning_rate
#         self.m_l1 =m_l1
#
#
#
#     def forward(self, X):
#
#         '''
#         forward propagation
#         :param X: input data
#         :returns: output data
#
#         '''
#
#         self.input = X
#         self.output = np.dot(X, self.weights) + self.bias
#         return self.output
#
#     def backward(self, d_out):
#
#         '''
#         backward propagation
#         :param d_out: derivative of output
#         :returns: derivative of input
#
#         '''
#         m = self.input.shape[0]
#         d_input = np.dot(d_out, self.weights.T)
#         dw = np.dot(self.input.T, d_out)/m + self.m_l1*np.sign(self.weights)
#         db = np.sum(d_out, axis=0, keepdims=True)/m
#         self.weights -= self.learning_rate * dw
#         self.bias -= self.learning_rate * db
#
#         return d_input

class Linear(Layer):
    def __init__(self, input_size, output_size, learning_rate=0.01):
        '''
        Linear layer
        :param
            input_size: input size
            output_size: output size
            learning_rate: learning rate
            m_l1: l1 regularization

        '''
        self.output = None
        self.input = None

        # he initialization
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.weights = np.concatenate([np.zeros((1, self.weights.shape[1])), self.weights], axis=0)

    def forward(self, X):
        '''
        forward propagation
        :param X: input data
        :returns: output data

        '''
        self.input = X
        # self.output = np.dot(X, self.weights) + self.bias
        self.X_b = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        self.output = np.dot(self.X_b, self.weights)

        return self.output

    def backward(self, d_out):
        '''
        backward propagation
        :param d_out: derivative of output
        :returns: derivative of input

        '''
        m = self.input.shape[0]
        d_input = np.dot(d_out, self.weights[1:, :].T)
        dw = np.dot(self.X_b.T, d_out) / m
        self.weights -= self.learning_rate * dw

        return d_input

class Dropout(Layer):
    def __init__(self, drop_rate=0.5):
        '''
        Dropout layer

        '''
        self.drop_rate = drop_rate
        self.mask = None

    def forward(self, X, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.drop_rate, size=X.shape) / (1 - self.drop_rate)
            return X * self.mask
        else:
            return X

    def backward(self, d_out):
        return d_out * self.mask
class Sigmoid(Layer):


    def __init__(self):
        self.output = None
        self.input = None

    def forward(self, X):
        self.input = X
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward(self, d_out):
        d_input = d_out * self.output * (1 - self.output)
        return d_input


class ReLU(Layer):

    def __init__(self):
        self.output = None
        self.input = None

    def forward(self, X):
        self.input = X
        self.output = np.maximum(0, X)
        return self.output

    def backward(self, d_out):
        d_input = d_out * (self.input > 0)
        return d_input


class BinaryCrossEntropyLoss(Layer):

    def __init__(self):
        self.true = None
        self.predict = None

    def forward(self, y_pred, y):
        # self.predict = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.predict = y_pred
        self.true = y
        loss = -np.mean(y * np.log(self.predict) + (1 - y) * np.log(1 - self.predict))
        return loss

    def backward(self):
        m = self.true.shape[0]
        # d_input = - (np.divide(self.true, self.predict) - np.divide(1 - self.true, 1 - self.predict)) /m
        d_input = np.divide(self.predict - self.true, self.predict * (1 - self.predict)) / m
        return d_input


class MeanSuaredErrorLoss(Layer):

    def __init__(self):
        self.loss = None
        self.true = None
        self.predict = None

    def forward(self, y_pred, y):
        self.predict = y_pred
        self.true = y
        loss = np.mean(np.square(self.predict - self.true)) / 2
        self.loss = loss
        return loss

    def backward(self):
        d_input = self.predict - self.true
        return d_input


class Sequential(Layer):

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):

        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def save(self, dir):
        pass


class L_layer_model_Classifier():

    def __init__(self, layer_dims, learning_rate=0.01, active_function='relu',random_state =666):
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = Sequential()
        L = len(layer_dims)
        if active_function == "relu":
            for i in range(L - 2):
                self.model.add_layer(Linear(layer_dims[i], layer_dims[i + 1], learning_rate))
                self.model.add_layer(ReLU())
        elif active_function == "sigmoid":
            for i in range(L - 2):
                self.model.add_layer(Linear(layer_dims[i], layer_dims[i + 1], learning_rate))
                self.model.add_layer(Sigmoid())
        self.model.add_layer(Linear(layer_dims[L - 2], layer_dims[L - 1], learning_rate))
        self.model.add_layer(Sigmoid())
        self.loss_function = BinaryCrossEntropyLoss()

    def fit(self, X, y, epochs=1000, print_cost=False):

        costs = []
        cost = None
        for epoch in range(epochs):

            y_pred = self.model.forward(X)
            cost = self.loss_function.forward(y_pred, y)
            d_loss = self.loss_function.backward()

            self.model.backward(d_loss)
            if epoch % 100 == 0 and print_cost:
                print(f"Epoch {epoch} : cost: {cost}")
            costs.append(cost)
        return cost

    def predict(self, X):
        AL = self.model.forward(X)

        if AL.shape[1] < 2:
            y_pred = (AL >= 0.5).astype(int)
        else:
            y_pred = np.argmax(AL)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        score = np.mean(y_pred == y)
        return score

    def __call__(self, X):
        return self.predict(X)


class L_layer_model_Regression():

    def __init__(self, layer_dims, learning_rate=0.01, active_function='relu',drop_rate=0,random_state=666):
        np.random.seed(random_state)
        self.random_state = random_state
        self.val_losses = []
        self.train_losses = []
        self.learning_rate = learning_rate
        self.model = Sequential()
        self.drop_rate =drop_rate
        L = len(layer_dims)

        if active_function == "relu":
            for i in range(L - 2):
                self.model.add_layer(Linear(layer_dims[i], layer_dims[i + 1], learning_rate))
                self.model.add_layer(ReLU())
                if drop_rate>0:
                    self.model.add_layer(Dropout(drop_rate))

        elif active_function == "sigmoid":
            for i in range(L - 2):
                self.model.add_layer(Linear(layer_dims[i], layer_dims[i + 1], learning_rate))
                self.model.add_layer(Sigmoid())
                if drop_rate > 0:
                    self.model.add_layer(Dropout(drop_rate))
        self.model.add_layer(Linear(layer_dims[L - 2], layer_dims[L - 1], learning_rate))
        self.loss_function = MeanSuaredErrorLoss()

    def fit(self, X, y, epochs=1000, patience=3, print_cost=False,epsilon=1e-10):

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=self.random_state)
        init_patience = patience
        y_val_pred = self.model.forward(X_val)
        val_loss = self.loss_function.forward(y_val_pred, y_val)

        for epoch in range(epochs):
            last_val_loss = val_loss

            # excute forward and backward fuction to update parameters
            y_train_pred = self.model.forward(X_train)
            train_loss = self.loss_function.forward(y_train_pred, y_train)
            d_loss = self.loss_function.backward()
            self.model.backward(d_loss)

            # calculate the loss of validation set
            y_val_pred = self.model.forward(X_val)
            val_loss = self.loss_function.forward(y_val_pred, y_val)

            # Stop training when the loss does not improve after patience steps
            if last_val_loss - val_loss < epsilon:
                init_patience -= 1
            else:
                init_patience = patience
            if epoch % 100 == 0 and print_cost:
                print(" after epoch %i: Train cost: %f, Validation cost: %f" % (epoch, train_loss, val_loss))

            if init_patience == 0:
                print(" after epoch %i: Train cost: %f, Validation cost: %f" % (epoch, train_loss, val_loss))

                break

            # save loss of train set and validation set
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

    def predict(self, X):
        y_pred = self.model.forward(X)


        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        score =mean_squared_log_error(abs(y),abs(y_pred))
        return score


    def __call__(self, X):
        return self.predict(X)
