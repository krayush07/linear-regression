# Least square method of multivariate linear regression.
# Solution/weight coeffecients = inv(X`X)X`y
# X -> 1 x1 x2 x3 ...
# https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


class LinearRegression:
    def __init__(self):
        self._X = None
        self._y = None
        self._correlation = []
        self._coeffecients = None

    def _mean(self, arr):
        return np.mean(arr)

    def _diff(self, arr, num):
        return np.subtract(arr, num)

    def fit(self, X, y):
        self._X = np.insert(X, 0, [1], axis=1)
        self._y = y
        self._compute_correlation()
        self._coeffecients = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self._X), self._X)), np.transpose(self._X)), self._y)  # inv(X'X)X'y
        train_pred = self.predict(X)
        return self._rmse_error(y, train_pred)

    def _rmse_error(self, y, y_pred):
        return np.sqrt(np.mean(np.subtract(y, y_pred) ** 2))

    def predict(self, input):
        input_feature = np.insert(input, 0, [1], axis=1)
        return np.dot(input_feature, self._coeffecients)

    def weight_coefficients(self, decimal=4):
        return self._coeffecients

    def _compute_correlation(self):
        sum_y = np.sum(self._y)
        n = len(self._y)
        for i in range(1, len(self._X[0]), 1):
            curr_feature = self._X[:, i]
            sum_x = np.sum(curr_feature)
            sum_xy = np.sum(np.multiply(curr_feature, self._y))
            sum_xsquare = np.sum(np.multiply(curr_feature, curr_feature))
            sum_ysquare = np.sum(np.multiply(self._y, self._y))

            self._correlation.append((n * sum_xy - sum_x * sum_y) / np.sqrt(((n * sum_xsquare - sum_x ** 2) * (n * sum_ysquare - sum_y ** 2))))

    def correlation(self, decimal=4):
        return self._correlation


def main():
    linear_reg = LinearRegression()

    X = np.genfromtxt('../resources/boston_housing/train_wo_col_labels.csv', dtype=np.float32, delimiter=',')
    X_train = X[:, 1:-1]
    y_train = X[:, -1]
    # X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    # y = [6, 8, 9, 11, 13, 16, 17, 19, 20, 24]

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_train = scaler.transform(X_train)

    rmse_error = linear_reg.fit(X_train, y_train)
    weights = linear_reg.weight_coefficients()
    print 'Weight coefficients for y = b0 + b1x1 + b2x2 + ... {} \n'.format(weights)
    print 'Correlation between X and y: {}\n'.format(linear_reg.correlation())
    print 'Root mean-squared error for training: {}\n'.format(rmse_error)

    # X_test = [[3], [5], [10]]
    X_test = np.genfromtxt('../resources/boston_housing/test_wo_col_labels.csv', dtype=np.float32, delimiter=',')
    X_test = X_test[:, 1:]
    X_test = scaler.transform(X_test)
    prediction = linear_reg.predict(X_test)
    # print 'Prediction of {}: {}'.format(X_test, (linear_reg.predict(X_test)))
    print 'Prediction on test instances: \n'
    for each_pred in prediction:
        print round(each_pred, 1)


if __name__ == '__main__':
    main()
