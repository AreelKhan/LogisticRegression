import numpy as np
from math import exp


def sigmoid(x):
    return 1 / (1 + exp(round(-x, 10)))


class LogisticRegression:

    def __init__(self):
        self.parameters = None

    def fit(self, X, y, logistic_function=sigmoid, learning_rate=0.1, epochs=3000):
        if len(X.shape) == 1:
            num_features = 1
        else:
            num_features = X.shape[1]

        function = np.vectorize(logistic_function)
        theta = np.ones(shape=num_features + 1)
        x_0 = np.ones_like(y)
        x = np.column_stack([x_0, X])
        for _ in range(epochs):
            cost = function(np.dot(x, theta)) - y
            theta = theta - (learning_rate * (1 / len(y)) * np.dot(x.T, cost))
        self.parameters = theta
        return None

    def pred(self, X):
        y_pred = []
        for i in range(len(X)):
            pred = 1 / (1 + exp(-np.dot(self.parameters, np.insert(X[i], 0, 1))))
            if pred > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred
