import numpy as np

def gaussianElimination(mat: np.ndarray, vector: np.ndarray) -> np.array:
    N = len(mat)
    
    mat = mat.tolist
    print(mat)
    for index, elem in enumerate(vector):
        mat[index].append(elem[0])
        # np.append(mat[index], elem[0])
    print(mat)
    return backSub(mat, N)
 

def swap_row(mat: np.array, i: int, j: int, N: int) -> np.array:
 
    for k in range(N + 1):
 
        temp = mat[i][k]
        mat[i][k] = mat[j][k]
        mat[j][k] = temp

    return mat
 

def forwardElim(mat: np.array, N: int) -> int:
    for k in range(N):
       
        i_max = k
        v_max = mat[i_max][k]
 
        for i in range(k + 1, N):
            if (abs(mat[i][k]) > v_max):
                v_max = mat[i][k]
                i_max = i
 
        if not mat[k][i_max]:
            return k
 
        if (i_max != k):
            mat = swap_row(mat, k, i_max, N)
 
        for i in range(k + 1, N):
            f = mat[i][k]/mat[k][k]
 
            for j in range(k + 1, N + 1):
                mat[i][j] -= mat[k][j]*f
 
            mat[i][k] = 0
 
    return -1
 
def backSub(mat: np.array, N: int) -> np.array:
 
    x = [None for _ in range(N)]

    for i in range(N-1, -1, -1):
        x[i] = mat[i][N]

        for j in range(i + 1, N):
            x[i] -= mat[i][j]*x[j]
 
        x[i] = (x[i]/mat[i][i])
 
    return x