def gaussianElimination(mat, vector, N):
 
    for index, elem in enumerate(vector):
        mat[index].append(elem[0])

    return backSub(mat, N)
 

def swap_row(mat, i, j, N):
 
    for k in range(N + 1):
 
        temp = mat[i][k]
        mat[i][k] = mat[j][k]
        mat[j][k] = temp
 

def forwardElim(mat, N):
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
            swap_row(mat, k, i_max, N)
 
        for i in range(k + 1, N):
            f = mat[i][k]/mat[k][k]
 
            for j in range(k + 1, N + 1):
                mat[i][j] -= mat[k][j]*f
 
            mat[i][k] = 0
 
    return -1
 
def backSub(mat, N):
 
    x = [None for _ in range(N)]

    for i in range(N-1, -1, -1):
        x[i] = mat[i][N]

        for j in range(i + 1, N):
            x[i] -= mat[i][j]*x[j]
 
        x[i] = (x[i]/mat[i][i])
 
    return x

if (__name__ == "__main__"):
    # Przykładowe macierze A i B
    matrix = [[10, -1, 2],
            [-1, 11, -1],
            [2, -1, 10]]
    vector = [[6], [25], [-11]]


    solution = gaussianElimination(matrix, vector, 3)
    print(solution)
