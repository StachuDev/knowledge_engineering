from typing import List
# from main import multiplyByScalar


def multiplyByScalar(v_1, s: float) -> float:
    v_2: List[float] = []
    for el in v_1:
        v_2.append(s*el)

    return v_2


def multiMatrixByScalar(matrix: List[List[float]], s: float) -> List[float]:
    result_v: List[float] = []
    for v in matrix:
        result_v.append(multiplyByScalar(v, s))

    return result_v


if __name__ == '__main__':
    m = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    s = 5

    output = multiMatrixByScalar(m, s)
    print(f'matrix: {m}')
    print(f'scalar: {s}')
    print(f'multiplication: {output}')
