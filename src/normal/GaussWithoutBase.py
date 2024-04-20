import numpy as np

def gauss_Elimination(matrix: np.ndarray, vector: np.ndarray):
    
    n = len(matrix)
    x = np.zeros(n)
    vector = list(map(lambda x: [x], vector))
    matrix = np.concatenate((matrix, vector), axis=1)

    for i in range(n):
        for j in range(i+1, n):
            ratio = matrix[j][i]/matrix[i][i]

            for k in range(n+1):
                matrix[j][k] = matrix[j][k] - ratio * matrix[i][k]

    x[n-1] = matrix[n-1][n]/matrix[n-1][n-1]

    for i in range(n-2,-1,-1):
        x[i] = matrix[i][n]

        for j in range(i+1,n):
            x[i] = x[i] - matrix[i][j]*x[j]

        x[i] = x[i]/matrix[i][i]

    np.savetxt("gaussianElimination.txt", np.transpose(x), fmt='%.4f')

