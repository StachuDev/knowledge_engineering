from typing import List

v1: List[float] = [1, 2, 3]
v2: List[float] = [7, 6, 9]

print(f'v1: {v1}, v2: {v2}')


def add(v_1: List[float], v_2: List[float]):
    if len(v_1) != len(v_2):
        print("The vectors have to have the same length!")
        return -1
    result: List[float] = []
    for el in range(len(v_1)):
        result.append(v_1[el] + v_2[el])

    return result


v3 = add(v1, v2)
print(f"add(v1, v2) = {v3}")


def multiply(v_1, v_2):
    if len(v_1) != len(v_2):
        print("The vectors have to have the same length!")
        return -1
    result: float = 0
    for el in range(len(v_1)):
        result += v_1[el] * v_2[el]

    return result


v4 = multiply(v1, v2)
print(f"multiply(v1, v2) = {v4}")
