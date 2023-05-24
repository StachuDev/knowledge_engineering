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


if __name__ == '__main__':
    m_1 = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4]
        ]
    )

    m_2 = np.array(
        [
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2]
        ]
    )

    m_r = multipMatrix(m_1, m_2)

    print(m_r)
