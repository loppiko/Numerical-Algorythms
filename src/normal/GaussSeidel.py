import numpy as np

def gauss_seidel(matrix, vector, x0=None, tol=1e-6, max_iter=1000):
    n = len(matrix)
    vector = vector.flatten()
    if x0 is None:
        x0 = [0] * n

    x = x0[:]
    for _ in range(max_iter):
        x_new = x[:]
        for i in range(n):
            sum_ = sum(matrix[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (vector[i] - sum_) / matrix[i][i]

        if sum((x_new[i] - x[i]) ** 2 for i in range(n)) ** 0.5 < tol:
            np.savetxt("gauss-Seidel.txt", x_new, fmt='%.4f')

        x = x_new