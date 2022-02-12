import numpy as np
from math import exp


def sigmoid(x):
    return 1 / (1 + exp(round(-x, 10)))


class LogisticRegression:

    def __init__(self):
        self.parameters = None

    def gradient_descent(self, train_data, labels, logistic_function=sigmoid, learning_rate=0.05, epochs=2000):
        if len(train_data.shape) == 1:
            num_features = 1
        else:
            num_features = train_data.shape[1]

        function = np.vectorize(logistic_function)
        theta = np.ones(shape=num_features + 1)
        x_0 = np.ones_like(labels)
        x = np.column_stack([x_0, train_data])
        for _ in range(epochs):
            cost = function(np.dot(x, theta)) - labels
            theta = theta - (learning_rate * (1 / len(labels)) * np.dot(x.T, cost))
        self.parameters = theta
        return theta
