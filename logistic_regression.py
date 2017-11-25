from activations import sigmoid

import numpy as np

class LogisticRegressionImpl:
    def __init__(self, alpha=5e-2, iterations=20000, regularization=10):
        self.alpha = alpha
        self.iterations = iterations
        self.regularization = regularization

    def train(self, train_X, train_Y):
        n, m = train_X.shape
        self.w = np.zeros((n, 1))
        self.b = 0

        for i in range(self.iterations):
            y_hat = self.__forward_propagation(train_X)
            cost = self.__compute_cost(y_hat, train_Y)
            self.__backward_propagation(y_hat, train_X, train_Y)
            if i % 10000 == 0:
                predictions = (y_hat > 0.5)
                accuracy = (m - np.sum(np.absolute(predictions - train_Y))) / m
                print("=== iteration {}, cost: {}, accuracy: {}".format(i, cost, accuracy))

        print("W = {}, b = {}".format(np.ndarray.flatten(self.w), self.b))

    def predict(self, X):
        a = self.__forward_propagation(X)
        return a > 0.5

    def __forward_propagation(self, x):
        return sigmoid(np.dot(self.w.T, x) + self.b)

    def __compute_cost(self, y_hat, y):
        _, m = y.shape
        cost = (-1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) + (self.regularization / (2 * m)) * np.sum(np.square(self.w))
        return np.squeeze(cost)

    def __backward_propagation(self, y_hat, x, y):
        _, m = x.shape
        dz = y_hat - y
        dw = np.dot(x, dz.T) / m + (self.regularization / m) * self.w
        db = np.sum(dz) / m
        self.w = self.w - self.alpha * dw
        self.b = self.b - self.alpha * db
