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

    def fit(self, X, y):
        self._X = X
        self._y = y
        self._compute_correlation()

        mean_x = self._mean(self._X)
        mean_y = self._mean(self._y)
        mean_deviation_x = self._diff(self._X, mean_x)
        mean_deviation_y = self._diff(self._y, mean_y)

        self._beta1 = np.sum(np.multiply(mean_deviation_x, mean_deviation_y)) / np.sum(mean_deviation_x ** 2)
        self._beta0 = mean_y - self._beta1 * mean_x

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
        plt.scatter(X, y, color="red", marker="o", s=30)

        y_pred = self._beta0 + np.multiply(self._beta1, X)

        plt.plot(X, y_pred, color="green")

        plt.xlabel('X')
        plt.ylabel('y')

        plt.show()


def main():
    linear_reg = LinearRegression()
    X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [6, 8, 9, 11, 13, 16, 17, 19, 20, 23]

    linear_reg.fit(X, y)
    weights = linear_reg.weight_coefficients()
    print 'Weight coefficients for y = mx + c: m = {}, c = {}\n'.format(weights[1], weights[0])
    print 'Correlation between X and y: {}\n'.format(linear_reg.correlation())

    test_y = [3, 5]
    print 'Prediction of {}: {}'.format(test_y, (linear_reg.predict(test_y)))
    linear_reg.plot_regression(X, y)


if __name__ == '__main__':
    main()
