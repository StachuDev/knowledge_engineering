from typing import List

v1: List[float] = [3, 2, 1]

s: float = 4


def multiplyByScalar(v_1, s: float) -> float:
    v_2: List[float] = []
    for el in v_1:
        v_2.append(s*el)

    return v_2


v3 = multiplyByScalar(v1, s)
print(v3)


