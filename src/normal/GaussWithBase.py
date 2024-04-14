def gauss_elimination_partial_pivot(matrix, vector):
    n = len(matrix)

    for i in range(n):
        # Wybór elementu podstawowego w kolumnie i
        max_row = i
        for k in range(i+1, n):
            if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                max_row = k
        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
        vector[i], vector[max_row] = vector[max_row], vector[i]

        # Eliminacja współczynników
        for j in range(i+1, n):
            factor = matrix[j][i] / matrix[i][i]
            vector[j] -= factor * vector[i]
            for k in range(i, n):
                matrix[j][k] -= factor * matrix[i][k]

    # Rozwiązanie układu równań
    solution = [0] * n
    for i in range(n - 1, -1, -1):
        solution[i] = vector[i]
        for j in range(i + 1, n):
            solution[i] -= matrix[i][j] * solution[j]
        solution[i] /= matrix[i][i]

    return solution