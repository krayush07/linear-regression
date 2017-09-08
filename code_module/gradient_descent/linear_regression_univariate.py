# Gradient descent method of univariate linear regression.

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self._X = None
        self._y = None
        self._beta0 = 0.
        self._beta1 = 0.
        self._correlation = 0.

    def _mean(self, arr):
        return np.mean(arr)

    def _diff(self, arr, num):
        return np.subtract(arr, num)

    def _run_epoch(self, X, y, num_epochs, lr):
        curr_error = 0.
        n = len(y)
        for i in range(num_epochs):
            curr_prediction = self.predict(X)
            curr_error = self._rmse_error(y, curr_prediction)
            beta0_grad = (2. / n) * np.sum(np.subtract(curr_prediction, y))
            beta1_grad = (2. / n) * np.sum(np.dot(np.subtract(curr_prediction, y), X))

            self._beta0 -= lr * beta0_grad
            self._beta1 -= lr * beta1_grad

            print('Loss at {} epoch: {}'.format((i + 1), round(curr_error, 5)))
        return curr_error

    def fit(self, X, y, num_epochs, lr, seed=np.random.randint(0, 1000000)):
        self._X = X
        self._y = y
        self._compute_correlation()
        np.random.seed(seed)
        self._beta0 = np.random.uniform(-0.1, 0.1)
        self._beta1 = np.random.uniform(-0.1, 0.1)
        rmse_error = self._run_epoch(self._X, self._y, num_epochs, lr)
        return rmse_error

    def _rmse_error(self, y, y_pred):
        return np.sqrt(np.mean(np.subtract(y, y_pred) ** 2))

    def predict(self, input):
        return np.add(self._beta0, np.dot(self._beta1, input))

    def weight_coefficients(self, decimal=4):
        return round(self._beta0, decimal), round(self._beta1, decimal)

    def _compute_correlation(self):
        sum_x = np.sum(self._X)
        sum_y = np.sum(self._y)
        sum_xy = np.sum(np.multiply(self._X, self._y))
        sum_xsquare = np.sum(np.multiply(self._X, self._X))
        sum_ysquare = np.sum(np.multiply(self._y, self._y))
        n = len(self._X)
        self._correlation = (n * sum_xy - sum_x * sum_y) / np.sqrt(((n * sum_xsquare - sum_x ** 2) * (n * sum_ysquare - sum_y ** 2)))

    def correlation(self, decimal=4):
        return round(self._correlation, decimal)

    def plot_regression(self, X, y):
        plt.scatter(X, y, color="red", marker="x", s=20)

        y_pred = self._beta0 + np.multiply(self._beta1, X)

        plt.plot(X, y_pred, color="green")

        plt.xlabel('X')
        plt.ylabel('y')

        plt.show()


def main():
    linear_reg = LinearRegression()
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([6, 8, 9, 11, 13, 16, 17, 19, 20, 24])

    rmse_error = linear_reg.fit(X, y, 200, 0.03, seed=12345)
    weights = linear_reg.weight_coefficients()
    print 'Weight coefficients for y = mx + c: m = {}, c = {}\n'.format(weights[1], weights[0])
    print 'Correlation between X and y: {}\n'.format(linear_reg.correlation())
    print 'Root mean-squared error for training: {}\n'.format(rmse_error)

    test_y = [3, 5, 10]
    print 'Prediction of {}: {}'.format(test_y, (linear_reg.predict(test_y)))
    linear_reg.plot_regression(X, y)


if __name__ == '__main__':
    main()
