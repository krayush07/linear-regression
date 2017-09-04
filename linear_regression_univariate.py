import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self._beta0 = 0.
        self._beta1 = 0.

    def _mean(self, arr):
        return np.mean(arr)

    def _diff(self, arr, num):
        return np.subtract(arr, num)

    def fit(self, X, y):
        mean_x = self._mean(X)
        mean_y = self._mean(y)
        mean_deviation_x = self._diff(X, mean_x)
        mean_deviation_y = self._diff(y, mean_y)

        self._beta1 = np.sum(np.dot(mean_deviation_x, mean_deviation_y)) / np.sum(np.dot(mean_deviation_x, mean_deviation_x))
        self._beta0 = mean_y - self._beta1 * mean_x

    def predict(self, input):
        return np.add(self._beta0, np.dot(self._beta1, input))

    def weight_coefficients(self):
        return self._beta0, self._beta1

    def plot_regression(self, X, y):
        plt.scatter(X, y, color="m", marker="o", s=30)

        y_pred = self._beta0 + np.multiply(self._beta1, X)

        plt.plot(X, y_pred, color="g")

        plt.xlabel('x')
        plt.ylabel('y')

        plt.show()

def main():
    linear_reg = LinearRegression()
    X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12]

    linear_reg.fit(X, y)
    print(linear_reg.predict([3, 5]))
    linear_reg.plot_regression(X, y)

if __name__ == '__main__':
    main()