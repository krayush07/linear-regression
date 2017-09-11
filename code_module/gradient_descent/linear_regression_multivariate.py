# Gradient Descent method of multivariate linear regression.

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

    def _run_epoch(self, X, y, theta, num_epochs, lr):
        curr_error = 0.
        n = len(y)
        loss = np.zeros(num_epochs)
        for i in range(num_epochs):
            curr_prediction = self.predict_train(X)
            curr_error = self._rmse_error(y, curr_prediction)
            loss[i] = curr_error
            error_diff = np.subtract(curr_prediction, y)
            theta[0] -= (lr/n) * np.sum(error_diff)

            for j in range(1, len(theta)-1, 1):
                theta[j] -= (lr/n) * np.sum(np.dot(error_diff, X[:,j]))

            print('Loss at {} epoch: {}'.format((i + 1), round(curr_error, 5)))
        self._plot_loss(loss)
        return curr_error

    def _plot_loss(self, loss):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(np.arange(len(loss)), loss, 'r')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost')
        ax.set_title('Error vs. Training Epoch')
        plt.show()

    def fit(self, X, y, num_epochs, lr, seed=np.random.randint(0, 1000000)):
        self._X = np.insert(X, 0, [1], axis=1)
        self._y = y
        self._compute_correlation()
        self._coeffecients = np.random.uniform(low=-0.001, high=0.001, size=len(self._X[0]))
        np.random.seed(seed)
        rmse_error = self._run_epoch(self._X, self._y, self._coeffecients, num_epochs, lr)
        # self._coeffecients = theta
        return rmse_error

    def _rmse_error(self, y, y_pred):
        return np.sqrt(np.mean(np.subtract(y, y_pred) ** 2))

    def predict_train(self, input):
        return np.dot(input, self._coeffecients)

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

    X = np.genfromtxt('../../resources/boston_housing/train_wo_col_labels.csv', dtype=np.float32, delimiter=',')
    X_train = X[:, 1:-1]
    y_train = X[:, -1]
    # X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    # y = [6, 8, 9, 11, 13, 16, 17, 19, 20, 24]

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_train = scaler.transform(X_train)

    rmse_error = linear_reg.fit(X_train, y_train, 5000, 0.01, seed=12345)
    weights = linear_reg.weight_coefficients()
    print 'Weight coefficients for y = b0 + b1x1 + b2x2 + ... {} \n'.format(weights)
    print 'Correlation between X and y: {}\n'.format(linear_reg.correlation())
    print 'Root mean-squared error for training: {}\n'.format(rmse_error)

    # X_test = [[3], [5], [10]]
    X_test = np.genfromtxt('../../resources/boston_housing/test_wo_col_labels.csv', dtype=np.float32, delimiter=',')
    X_test = X_test[:, 1:]
    X_test = scaler.transform(X_test)
    prediction = linear_reg.predict(X_test)
    # print 'Prediction of {}: {}'.format(X_test, (linear_reg.predict(X_test)))
    print 'Prediction on test instances: \n'
    # for each_pred in prediction:
    #     print round(each_pred, 1)


if __name__ == '__main__':
    main()
