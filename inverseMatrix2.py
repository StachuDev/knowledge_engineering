import numpy as np
from typing import Tuple, List


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
    

def substract_row(matrix: np.ndarray, index_1: int, index_2: int, times: int = 1) -> None:
        row_2 = matrix[index_2, :].copy()
        matrix[index_1, :] = matrix[index_1, :] - row_2*times


def divide_row(matrix: np.ndarray, index: int, value: float) -> None:
    matrix[index, :] = matrix[index, :] / value


# swap rows: matrix[index_1, :] and matrix[index_2, :]
def reorder_row(matrix: np.ndarray, index_1: int, index_2: int):
    row_1 = matrix[index_1, :].copy()
    matrix[index_1, :] = matrix[index_2, :]
    matrix[index_2, :] = row_1


def to_bottom(matrix: np.ndarray, index_1: int):
    row_1 = matrix[index_1, :].copy()
    # gdy zrobie matrix = np.delete(matrix, index_1, axis=0)
    # zostanie utworzona kopia matrix dlatego robie inaczej:
    matrix[:matrix.shape[0]-1] = np.delete(matrix, index_1, axis=0)
    matrix[matrix.shape[0]-1, :] = row_1
    


class Matrix:
    matrix: np.ndarray
    shape: Tuple[int, int]

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix.astype('float64')
        self.shape = matrix.shape

    def show(self):
        print(self.matrix)

    def inverse_matrix_gaussian_elimination(self) -> np.ndarray:
        # macierz jednostkowa
        matrix_copy = self.matrix.copy()
        identity_matrix = np.identity(self.shape[0]).astype('float64')

        def from_triangle_matrix_to_identity(matrix: np.ndarray, ident_matrix: np.ndarray):
            start_row_iter = matrix.shape[0]-2
            # pętla przechodzi po wszystkich elementach od dolnego prawego rogu
            for col_index in range(matrix.shape[0]-1, 0, -1):
                
                for row_index in range(start_row_iter, -1, -1):
                    value = matrix[row_index, col_index]

                    # "value" razy odejmuje od biezacego wiersza wiersz ostatni
                    substract_row(matrix, row_index, col_index, value)
                    substract_row(ident_matrix, row_index, col_index, value)
                # kolejną kolumne zaczynamy o jeden wiersz wyżej
                start_row_iter -= 1
                

        # def reset_column_using_first_row(matrix: np.ndarray, col_index, ident_matrix: np.ndarray) -> None:
        #     row_iter: int = col_index + 1
        #     while row_iter < matrix.shape[0] and matrix[row_iter, col_index] != 0:
        #         substract_row(matrix, row_iter, col_index)
        #         substract_row(ident_matrix, row_iter, col_index)
        #         row_iter += 1

        
        temp = 0
        for col_index in range(self.shape[0]):
            first_time = True
            template_index = temp
            zeros_count = 0
            row_index = temp
            
            for row_index in range(temp, self.shape[0]):
                

                row_index = row_index - zeros_count
                # jeli wartosc w pierwszej kolumnie jest różna od zera
                if matrix_copy[row_index, col_index] != 0:  # 1 != 0

                    value = matrix_copy[row_index, col_index]

                    # łapiemy pierwszą wartość w kolumnie 
                    if first_time is True:
                        # dzielimy tę wartość aby była równa jeden
                        divide_row(matrix_copy, row_index, value)
                        divide_row(identity_matrix, row_index, value)
                        # ustawiamy jej index jako szablon
                        template_index = row_index
                        first_time = False
                    else:
                        # kożystając z szblonu odpowiednią ilość razy odejmujemy go
                        # aby wyzerować pozostałe wiersze
                        substract_row(matrix_copy, row_index, template_index, value)
                        substract_row(identity_matrix, row_index, template_index, value)
                # wiersz zaczynający się od zera (w danej kolumnie)                  
                else:
                    # przerzucamy go na sam dół, 
                    # pozostałe przesuwają się o jeden do góry 
                    to_bottom(matrix_copy, row_index)
                    to_bottom(identity_matrix, row_index)
                    zeros_count += 1
            
            # matrix_copy = matrix_copy.round(3)

            temp += 1

        # from_triangle_matrix_to_identity(matrix_copy, identity_matrix)

        # return identity_matrix
        return matrix_copy  
 
            
if __name__ == '__main__':

    m1 = np.asarray([
        [1,2,3,7,1],
        [9,2,5,1,2],
        [0,3,2,2,9],
        [1,1,0,1,1],
        [3,7,6,2,1]
        ])
    
    m2 = np.asarray([
        [1,2,3,7,1],
        [9,2,5,1,2],
        [0,3,2,2,9],
        [1,1,0,1,1],
        [1,1,0,1,1]
        ])
    
    v1: np.ndarray = np.asarray([
        [0, 4, 5],
        [9, 1, 0],
        [2, 3, 1]
    ])



    matrix = Matrix(m2)

    
    print(matrix.inverse_matrix_gaussian_elimination())
    
    


    # matrix2 = Matrix(m1)

    # m1_invers = matrix2.inverse_matrix_gaussian_elimination()

    # identity2 = multipMatrix(m1, m1_invers)

    # print('matrix B:')
    # print(v1)
    # print('matrix B^-1:')
    # print(m1_invers)
    # print('result B * B^-1:')
    # print(identity2) 
    # print()
    # print('Correct result: ')
    # inverse = np.linalg.inv(m1)
    # print(inverse)

    # print()
    # print(abs(inverse - m1_invers))
