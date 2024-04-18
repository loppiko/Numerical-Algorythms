import numpy as np

import numpy as np

def gauss_elimination_with_partial_pivot(matrix, vector):
    A = np.array(matrix, dtype=float)
    f = np.array(vector, dtype=float)
    length = f.size
    for i in range(0, length - 1):     
        for j in range(i + 1, length):
            if A[i, i] == 0:
                break
            m = A[j, i] / A[i, i]
            A[j, :] = A[j, :] - m * A[i, :]
            f[j] = f[j] - m * f[i]
    np.savetxt("gaussEliminationPivot.txt", np.transpose(Back_Subs(A, f)), fmt='%.2f')

def Back_Subs(A, f):
    length = f.size
    x = np.zeros(length)
    x[length - 1] = f[length - 1] / A[length - 1, length - 1]
    for i in range(length - 2, -1, -1):
        sum_ = 0
        for j in range(i + 1, length): 
            sum_ = sum_ + A[i, j] * x[j]
        x[i] = (f[i] - sum_) / A[i, i]
    return x