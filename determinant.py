import numpy as np

def determinant(size: int, matrix: np.ndarray) -> float:
    sign: int
    result: float
    new_matrix: np.ndarray
    if size == 1:
        # gdy maciez ma wymiar 1x1 zwracamy wartosc jednej komorki
        return int(matrix)
    else:
        sign = 1
        result = 0

        # przechodzimy kolejno po kazdym indeksie pierwszego wiersza macierzy
        for i in range(size):
            # tworzymy minor dla kazdego z tych indeksow przez:
            # usuniecie pierwszego wiersza
            new_matrix = np.delete(matrix, 0, 0)

            # usuniecie koljnych kolumn
            new_matrix = np.delete(new_matrix, i, 1)

            result += sign * determinant(size-1, new_matrix) * matrix[0, i]

            # ustawia nam znak
            sign = -sign

        return result


if __name__ == '__main__':

    m = np.arange(9).reshape(3, 3)
    m = m+1
    # print(m)
    # print(" ")
    # print(determinant(3, m))

    a_1 = np.asarray(
        [
            [1, 2, 3],
            [6, 5, 4],
            [3, 7, 2]
        ]
    )

    a_2 = np.asarray(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    )

    a_3 = np.asarray(
        [
            [1,2,1],
            [1,5,6],
            [3,8,1]
        ]
    )

    print(int(determinant(3, a_1)) == 63)
    print(int(determinant(3, a_2)) == 0)
    print(int(determinant(3, a_3)) == -16)
    # print(" ")
    # print(a)
    # print(" ")
    # print(determinant(3, a))



