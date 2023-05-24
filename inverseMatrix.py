import numpy as np


def multiply(v_1, v_2):
    if len(v_1) != len(v_2):
        print("The vectors have to have the same length!")
        return -1
    result: float = 0
    for el in range(len(v_1)):
        result += v_1[el] * v_2[el]

    return result


def multipMatrix(matrix_1: np.ndarray, matrix_2: np.ndarray):
    shape_1 = matrix_1.shape
    shape_2 = matrix_2.shape

    if shape_1[1] == shape_2[0]:
        result_shape = (shape_1[0], shape_2[1])
        result_matrix = np.ones(result_shape, dtype=float)
        # go across first matrix row
        for row_iter in range(shape_1[0]):
            # go across second matrix column
            for col_iter in range(shape_2[1]):
                result_matrix[row_iter][col_iter] = multiply(matrix_1[row_iter, :], matrix_2[:, col_iter])
    else:
        return -1

    return result_matrix


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


# macierz dopełnień algebraicznych
# det(matrix) != 0
# matrix is square
def matrix_of_minors(matrix):
    size = matrix.shape[0]
    result_matrix = np.ones(matrix.shape, dtype=float)
    sign: int = 1
    for row_iter in range(size):

        # usuniecie bieżącego wiersza
        minor = np.delete(matrix, row_iter, 0)
        for col_iter in range(size):
            # usuniecie bieżącej kolumny
            new_minor = np.delete(minor, col_iter, 1)

            result_matrix[row_iter][col_iter] = sign*determinant(size-1, new_minor)

            # ustawia nam znak
            sign = -sign

    return result_matrix


def transpose(matrix: np.ndarray):
    shape = matrix.shape
    shape_t = (shape[1], shape[0])
    matrix_t = np.ones(shape_t, dtype=float)
    for row_iter in range(shape[0]):
        for col_iter in range(shape[1]):
            matrix_t[col_iter][row_iter] = matrix[row_iter][col_iter]

    return matrix_t


# macierz musi byc kwadratowa
def inverse_matrix(size: int, matrix: np.ndarray):
    transpose_matrix_of_minors = transpose(matrix_of_minors(matrix))
    result_matrix: np.ndarray
    det = determinant(size, matrix)
    if det == 0:
        return -1

    else:
        result_matrix = transpose_matrix_of_minors / det
        return result_matrix


if __name__ == '__main__':

    a = np.asarray(
        [
            [1, 2, 3],
            [6, 5, 4],
            [3, 7, 2],
        ]
    )

    i_a = inverse_matrix(3, a)
    print(i_a)

    # test
    print(determinant(3, a))
    r_a = multipMatrix(i_a, a)
    print(r_a)
    
    
    #
    # c_m = complement_matrix(a)
    # print(c_m)
