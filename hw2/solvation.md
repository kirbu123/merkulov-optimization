# Conditional gradient methods

### Task 1 

$\nabla f(X_k)$ - ?

$f(X) = \frac{1}{2} || X - Y ||_F^2$

тогда $f(X) = \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} X_{ij}^2$, $\nabla f(X) = X - Y$




Решение задачи LMO

Линейная задача минимизации (LMO):

$\min \langle \nabla f(X_k), S \rangle_{S \in B_n} = \min \langle X_k - Y, S \rangle_{S \in B_n} = \min tr((X_k - Y)^T S)_{S \in B_n}$

Задача сводится к:

$\min tr((X_k^T-Y^T) S)_{S \in B_n}$

Минимизация $tr((X_k^T-Y^T) S)$ при $S \in B_n$ — это задача линейного назначения (linear assignment problem).

Применим Венгерский алгоритм к матрице $X_k-Y$, чтобы найти оптимальное назначение.

Решение $S_k$, будет перестановочной матрицей, соответствующей оптимальному назначению.

### Task 2

See implementation in ```solvation.ipynb```

### Task 
See test code in ```solvation.ipynb```