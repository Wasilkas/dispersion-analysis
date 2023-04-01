import numpy as np
from copy import deepcopy


class DispersionAnalysisModel:
    def __init__(self, data: np.ndarray, factor_levels_points: list, levels: list, k: int) -> None:
        self.__data = deepcopy(data)
        self.__n = len(factor_levels_points)
        self.__m = sum(levels)
        self.__x = np.zeros(shape=(self.__n, self.__m))
        self.__levels = deepcopy(levels)
        self.__k = k

        for index, point in enumerate(factor_levels_points):
            self.__x[index, 0] = 1

            for i in range(k):
                self.__x[index, point[i] + sum(levels[:(i+1)]) - 1] = 1

        self.__reduced_x = self.__reduce_model()

    def __reduce_model(self) -> np.ndarray:
        levels = self.__levels
        x = self.__x
        reduced_x = np.zeros((self.__n, self.__m - self.__k))
        reduced_x[:, 0] = deepcopy(x[:, 0])

        for i in range(self.__k):
            reduced_x[:, 0] += x[:, sum(levels[:(i+2)]) - 1]

            for j in range(levels[i + 1] - 1):
                reduced_x[:, levels[i] + j] = x[:, levels[i] + j + i] - x[:, sum(levels[:(i + 2)]) - 1]
        x = self.__x
        return reduced_x

    def fit(self):
        r_x = self.__reduced_x
        return np.linalg.pinv(r_x.T @ r_x) @ r_x.T @ self.__data

    def fitting_terms(self, theta):
        return self.__reduced_x @ theta


def main():
    y = np.array([3.1, 2.9, 3.2, 4., 2.1, 1.9, 2.1, 1.9, 3.9, 4.1, 4.9, 5.1, 2.9, 3.1, 3.2, 3.0, 3.0, 2.1, 1.9, 1.95, 3.1, 2.9, 1., 1., 1., 0.95])
    levels_points = [[1, 1],
                     [1, 1],
                     [1, 1],
                     [1, 2],
                     [1, 3],
                     [1, 3],
                     [1, 4],
                     [1, 4],
                     [2, 1],
                     [2, 1],
                     [2, 2],
                     [2, 2],
                     [2, 3],
                     [2, 3],
                     [2, 3],
                     [2, 4],
                     [2, 4],
                     [3, 1],
                     [3, 1],
                     [3, 1],
                     [3, 2],
                     [3, 2],
                     [3, 3],
                     [3, 4],
                     [3, 4],
                     [3, 4]]
    model = DispersionAnalysisModel(y[:, np.newaxis], levels_points, [1, 3, 4], 2)
    theta = model.fit()
    print(theta)


if __name__ == '__main__':
    main()
