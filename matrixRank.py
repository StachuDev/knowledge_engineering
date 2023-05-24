import numpy as np
from typing import Tuple

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
    

def count_linearly_independent_rows(matrix: np.ndarray):
            matrix_sum = matrix.sum(1)
            row_count = 0
            for el in matrix_sum:
                if el.round(5) != 0:
                    row_count += 1

            return row_count

def matrix_rank(matrix: np.ndarray) -> int:
        matrix = matrix.astype('float64')
        shape = matrix.shape
        temp = 0

        for col_index in range(shape[0]):
            first_time = True
            template_index = temp
            zeros_count = 0
            row_index = temp
            
            for row_index in range(temp, shape[0]):
                row_index = row_index - zeros_count

                # jeli wartosc w pierwszej kolumnie jest różna od zera
                if matrix[row_index, col_index] != 0:  # 1 != 0
                    value = matrix[row_index, col_index]

                    # łapiemy pierwszą wartość w kolumnie 
                    if first_time is True:
                        # dzielimy tę wartość aby była równa jeden
                        divide_row(matrix, row_index, value)
                        # ustawiamy jej index jako szablon
                        template_index = row_index
                        first_time = False

                    else:
                        # kożystając z szblonu odpowiednią ilość razy odejmujemy go
                        # aby wyzerować pozostałe wiersze
                        substract_row(matrix, row_index, template_index, value)

                # wiersz zaczynający się od zera (w danej kolumnie)                  
                else:
                    # przerzucamy go na sam dół, 
                    # pozostałe przesuwają się o jeden do góry 
                    to_bottom(matrix, row_index)
                    zeros_count += 1

            temp += 1

        result = count_linearly_independent_rows(matrix)
        
        return result  
 
            
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
    
    m3 = np.asarray([
        [1,2,3,7,1],
        [9,2,5,1,2],
        [0,3,2,2,9],
        [2,4,6,14,2],
        [18,4,10,2,4],
        ])
    
    m4 = np.asarray([
        [1,2,3,0,1],
        [9,2,5,0,2],
        [0,3,2,0,9],
        [2,4,6,0,2],
        [18,4,10,0,4],
        ])
    
    print(matrix_rank(m1) == 5)

    print(matrix_rank(m2) == 4)

    print(matrix_rank(m3) == 3)

    print(matrix_rank(m4) == 3)    