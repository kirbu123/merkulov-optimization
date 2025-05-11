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

### Task 3
See test code in ```solvation.ipynb```

# Subgradient method

Task 4

See test code in ```solvation.ipynb```

See page 16 of presentation 18 for method convergence evidence.

Set loss function: $f(x) = ||A^{\frac{1}{2}} (x-y)||_2 - 1 + ||\Sigma x||_{\inf} - 1$

Gradient of loss function: $\nabla f(x) = 2 (A^{\frac{1}{2}})^TA^{\frac{1}{2}}(x - y) + \nabla ||\Sigma x||_{\inf}$

Where $\nabla f(x) =$ $
\begin{bmatrix}
0 \\
... \\
\sigma_{max} \\
... \\
\end{bmatrix}
$

# Proximal gradient method




### Subgradient Method

For a non-smooth convex function $f(W)$, the subgradient update at step $k$ is:

$ W_{k+1}=W_k−α_kg_k$ 


where:

$ α_k > 0 $ is the step size.

$ g_k∈∂f(W_k) $ is any subgradient of $f$ at $W_k$​.

Where:

$\nabla \| W \|_1 = \text{sign}(W)$,

  $$(\nabla W)_{ij} = \text{sign}(W)_{ij} = 
  \begin{cases} 
    +1 & \text{if } W_{ij} > 0, \\ 
    -1 & \text{if } W_{ij} < 0, \\ 
    \text{any } value \in [-1, 1] & \text{if } W_{ij} = 0 \ \text{(typically } 0\text{)}
  \end{cases}$$

# Proximal Gradient Method

For a composite function $ f(W)=g(W)+h(W) $, where $g$ is convex and differentiable, and $h$ is convex but non-smooth, the update is:

$ W_{k+1}=prox_{α_kh}(W_k−α_k∇g(W_k)) $

where:

$α_k ​> 0$ is the step size.

$ prox_{αh}​(V)=argmin_W​(h(W)+\frac{1}{2 \alpha}​∥W−V∥_2^2​) $ is the proximal operator of $h$.

See code in ```solvation.ipynb```

# Stochastic gradient methods

Общий вывод:
На сильно выпуклых функциях (MSE с L2L2​):

    1) SAG и SVRG сходятся линейно, обгоняя SGD.

    2) SVRG предпочтительнее из-за экономии памяти.

На выпуклых, но не сильно выпуклых (LogLoss без регуляризации):

    1) SAG может быть нестабилен (если μ≈0).

    2) SVRG всё ещё хорош, но требует аккуратного выбора частоты обновления полного градиента.

    3) SGD сходится, но медленно и с большим разбросом.

See code && report in ```solvation.ipynb```

# Big Models

| Setup | # of parameters | GPU peak memory, MB | Final eval loss | Batch Size | Time to run 5 epochs, s | Generation example | Comment |
|:---:|:---:|:---:|:---:|:---:|:---:|:---------:|:---------:|
| Baseline (GPT2) | 124 M | 10101 | 2.126 | 8 | 377.29 | `A long time ago in a galaxy far far away... there was a little girl named Lina. She was very curious and wanted to explore. One day, she was walking in the forest when she saw a big tree. She looked up and saw a big, shiny tree. She was so excited! She jumped up and ran to the tree, but it was too big. Lina was so excited and she ran to the tree and jumped up. She ran to the tree` | |
| facebook/opt-125m | 125 M | 6753 | 1.825 | 8 | 365.27 | `A long time ago in a galaxy far far away... there was a little girl named Lucy. She was three years old and she loved to play with her toys. One day, she decided to go to the park with her mom. She went to the park and saw a big tree. She looked around and saw a big tree with lots of flowers. She was so excited to see the big tree. She went to the tree and saw a big flower. She was so happy that she could` | |
| facebook/opt-125m | 125 M | 4233 | 1.745 | 4 | 341.27 | `A long time ago in a galaxy far far away... there was a little girl named Lucy. Lucy was very curious and wanted to know what was happening. She asked her mom, "What is happening here?" Her mom smiled and said, "It's a big galaxy. It's very big and very bright. It's very dark and very dark. It's very scary."Lucy was scared and wanted to know what was happening. She asked her mom, "What` | |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

See code in ```solvation.ipynb```

