import numpy as np
from typing import List, Tuple


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


class Matrix:
    matrix: np.ndarray
    shape: Tuple[int, int]

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix.astype('float64')
        self.shape = matrix.shape

    def show(self):
        print(self.matrix)


    def solve_equation(self, results_matrix) -> np.ndarray:
        matrix_copy = self.matrix.copy()

        def from_triangle_matrix_to_identity(matrix: np.ndarray, results_matrix: np.ndarray):
            start_row_iter = matrix.shape[0]-2
            # pętla przechodzi po wszystkich elementach od dolnego prawego rogu
            for col_index in range(matrix.shape[0]-1, 0, -1):
                
                for row_index in range(start_row_iter, -1, -1):
                    value = matrix[row_index, col_index]

                    # "value" razy odejmuje od biezacego wiersza wiersz ostatni
                    substract_row(matrix, row_index, col_index, value)
                    substract_row(results_matrix, row_index, col_index, value)
                # kolejną kolumne zaczynamy o jeden wiersz wyżej
                start_row_iter -= 1
                

        def reset_column_using_first_row(matrix: np.ndarray, col_index, results_matrix: np.ndarray) -> None:
            row_iter: int = col_index + 1
            while row_iter < matrix.shape[0] and matrix[row_iter, col_index] != 0:
                substract_row(matrix, row_iter, col_index)
                substract_row(results_matrix, row_iter, col_index)
                row_iter += 1

        temp = 0
        for col_index in range(self.shape[0]):
            
            # przechodzimy po każdym wierszu
            for row_index in range(temp, self.shape[0]):

                # jeli wartosc w pierwszej kolumnie jest różna od zera
                if matrix_copy[row_index, col_index] != 0:  # 1 != 0

                    # zapisujemy tą wartość z pierwszej kolumny
                    first_value = matrix_copy[row_index, col_index]

                    # dzielimy cawły bieżący wiersz przez tą wartość
                    divide_row(matrix_copy, row_index, first_value)
                    divide_row(results_matrix, row_index, first_value)
                else:
                    # zamieniamy miejscami bieżącym wiersz z ostatnim 
                    reorder_row(matrix_copy, self.shape[0]-1, row_index)
                    reorder_row(results_matrix, self.shape[0]-1, row_index)
            
            # matrix_copy = matrix_copy.round(3)

            reset_column_using_first_row(matrix_copy, col_index, results_matrix=results_matrix)
            temp += 1
        
        from_triangle_matrix_to_identity(matrix_copy, results_matrix=results_matrix)

        return results_matrix.round(3)



# def read_equations(list):
#     equation_matrix: np.ndarray = np.ones()
#     for str in list:
#         for index in range(len(str)):
#             if str[index] == " ":

if __name__ == '__main__':


    # var_count = int(input("Enter number of variables " +
    #       "(This number must be the same as number of equations):"))
    
    # print(var_count)

    # print("Next, enter your equations")
    # print("Important! Syntax: numberX + numberY + numberZ + ... = number")
    # print("example: 1x + 0z - 3y = 7")

    # equations_list = []
    # for i in range(var_count):
    #     equ = input(f"enter equation number: {i}\n")
    #     equations_list.append(equ)

    # list = ["x + 2y + 3z = 5", "1x + 22y + 3z = 51", "1x + 2y + 33z = 5"]
    # read_equations(list)

    # l = [
    #     [1, 2, 3],
    #     [1, 2, 3],
    #     [1, 2, 3]
    # ]
    # print(type(np.asarray(l)))

    arr = []
    result_list = []
    print("wprowadź liczby przed zmiennymi w dwóch równaniach: ")
    for i in range(2):
        list = []
        num_x = int(input("przed x:"))
        num_y = int(input("przed y:"))
        result = int(input("wynik: "))
        list.append(num_x)
        list.append(num_y)
        result_list.append(result)
        arr.append(list)
    
    arr = np.asarray(arr)
    result_list = np.asarray(result_list)[:, np.newaxis]

    matrix = Matrix(arr)
    matrix.show()

    result_arr = matrix.solve_equation(results_matrix=result_list)
    print(result_arr)


    


    
    


    
    