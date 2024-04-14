def gauss_elimination(A, B):
    rows = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])

    # Eliminacja
    for i in range(rows):
        # Pivotowanie, jeśli to konieczne
        if A[i][i] == 0:
            return "Algorytm nie może być zastosowany ze względu na dzielenie przez zero."

        for j in range(i+1, rows):
            factor = A[j][i] / A[i][i]
            for k in range(cols_A):
                A[j][k] -= factor * A[i][k]
            for k in range(cols_B):
                B[j][k] -= factor * B[i][k]

    # Rozwiązanie
    solution = [[0] * cols_B for _ in range(rows)]
    for i in range(rows - 1, -1, -1):
        for j in range(cols_B):
            solution[i][j] = B[i][j] / A[i][i]
        for j in range(i - 1, -1, -1):
            for k in range(cols_B):
                B[j][k] -= A[j][i] * solution[i][k]

    return solution
