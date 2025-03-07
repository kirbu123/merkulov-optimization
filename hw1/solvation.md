# Gradient Descent

We will assume mothing about the convexity of $f$. We will show that gradient descent reaches an $\epsilon$-substationary point $x$, such that $\|\nabla f(x)\|_2 \leq \epsilon$, in $O(1/\epsilon^2)$ iterations.

Lets write Lipschitz parabolic upper bound:

\[
f(y) \leq f(x) + \nabla f(x)^T (y - x) + \frac{L}{2} \|y - x\|_2^2, \quad \text{for all } x, y. \quad (1)
\]

Lets plug in $y = x^{k + 1} = x ^ k - \alpha \nabla f(x^k), x = x^k$ to equation $(1)$

\[
f(x^{k + 1}) \leq f(x^k) + \nabla f(x^k) (-\alpha \nabla f(x^k)) + \frac{L}{2}\alpha^2 \| \nabla f(x^k) \|_2^2
\]

\[
f(x^{k+1}) \leq f(x^k) - \alpha \| \nabla f(x^k) \|_2^2 + \frac{\alpha^2L}{2}\|\nabla f(x^k) \|_2^2
\]

\[
f(x^{k+1}) \leq f(x^k) + \alpha \| \nabla f(x^k) \|_2^2(\frac{\alpha L}{2} - 1)
\]

Lets use $\alpha \leq 1/L$, and rearrange the previous result:

\[
\alpha (1 - \frac{L\alpha}{2}) \| \nabla f(x^k) \|_2^2 \leq f(x^k) - f(x^{k + 1})
\]

\[
\frac{\alpha}{2} \| \nabla f(x^k) \|_2^2 \leq \alpha (1 - \frac{L\alpha}{2}) \| \nabla f(x^k) \|_2^2 \leq f(x^k) - f(x^{k + 1})
\]

\[
\| \nabla f(x^k) \|_2^2 \leq \frac{2}{\alpha} (f(x^k) - f(x^{k + 1}))
\]

Lets sum the previous result over all iterations from $1,\ldots,k+1$:

\[
\sum_{i=0}^{k} \| \nabla f(x^i) \|_2^2 \leq \frac{2}{\alpha} (f(x^0) - f(x^1) + f(x^1) - f(x^2) + f(x^2) + ...) = \frac{2}{\alpha} (f(x^0) - f(x^*))
\]

Lets lower bound the sum in the previous result to get:

\[
k \cdot \min_{i=0, \dots, k} \|\nabla f(x^i)\|_2 \leq \sqrt{\sum_{i=0}^{k} \| \nabla f(x^i) \|_2^2} \leq
 \sqrt{\frac{2}{\alpha (k+1)} (f(x^0) - f^*)}
\]

\[
\min_{i=0, \dots, k} \|\nabla f(x^i)\|_2 \leq \sqrt{\frac{2}{\alpha (k+1)} (f(x^0) - f^*)}
\]
