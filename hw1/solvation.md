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

# Accelerated methods
1) 
To solve this task, we need to analyze the **local convergence** of the **Heavy Ball Method** applied to the given function. The function \( f(x) \) is piecewise quadratic, meaning its gradient \( \nabla f(x) \) is piecewise linear. Since the method is known to perform well for strongly convex quadratics using the optimal hyperparameters:

\[
\alpha^* = \frac{4}{(\sqrt{L}+\sqrt{\mu})^2}, \quad \beta^* = \left(\frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}\right)^2
\]

Given:

\[
f(x) =
\begin{cases}
\frac{25}{2}x^2, & \text{if } x < 1 \\
\frac{1}{2}x^2 + 24x - 12, & \text{if } 1 \leq x < 2 \\
\frac{25}{2}x^2 - 24x +36, & \text{if } x \geq 2
\end{cases}
\]

and its derivative:

\[
\nabla f(x) =
\begin{cases}
25x, & \text{if } x < 1 \\
x + 24, & \text{if } 1 \leq x < 2 \\
25x - 24, & \text{if } x \geq 2
\end{cases}
\]

Lets prove, that the given function is convex, strongly convex, smooth.

Lets prove $\mu$-strong convexity and find coefficient.

$\mu$-strong means, that $\forall x, y, \forall \lambda \in [0, 1]:$


\[
f(y) \geq f(x) + \nabla f(x) (y - x) + \frac{\mu}{2} \| y - x \|^2
\]

\[
f(\lambda x_1 + (1 - \lambda) x_2) \leq \lambda f(x_1) + (1 - \lambda) f(x_1) - \frac{\mu}{2}\lambda (1-\lambda)\| x_1 - x_2 \|^2
\]

for function $g(x) = ax^2 + bx + c$: $\mu = 2a \Rightarrow$ for $f(x) \; \mu=2 \cdot min(\frac{25}{2}, \frac{1}{2}, \frac{25}{2}) = 2 \cdot \frac{1}{2} = 1$

Lets count $L$, that will prove smoothness of function $f = f(x)$

\[
\forall x, y: \; \| \nabla f(x) - \nabla f(y) \| \leq L\| x - y \|
\]

\[
L = max(\nabla^2f) = max(25, 1, 25) = 25
\]

Now, we are plotting the function value for $x \in [-4, 4]$ (look in the .ipynb report)

\[
\nabla f(x) = \frac{1}{m} \sum_{i=1}^{m} -b_i (1 - \sigma(b_i \langle a_i, x \rangle)) a_i + \lambda x
\]



\[
\nabla f(x) = -\frac{1}{m} \sum_{i=1}^{m} b_i (1 - \sigma(b_i \langle a_i, x \rangle)) a_i + \lambda x
\]

2.

Lets assume classification task:

Logistic regression is a standard model in classification tasks. For simplicity, consider only the case of binary classification. Informally, the problem is formulated as follows: There is a training sample $\{(a_i, b_i)\}_{i=1}^m$, consisting of $m$ vectors $a_i \in \mathbb{R}^n$ (referred to as features) and corresponding numbers $b_i \in \{-1, 1\}$ (referred to as classes or labels). The goal is to construct an algorithm $b(\cdot)$, which for any new feature vector $a$ automatically determines its class $b(a) \in \{-1, 1\}$.
In the logistic regression model, the class determination is performed based on the sign of the linear combination of the components of the vector $a$ with some fixed coefficients $x \in \mathbb{R}^n$:
$$
b(a) := \text{sign}(\langle a, x \rangle).
$$
The coefficients $x$ are the parameters of the model and are adjusted by solving the following optimization problem:
$$
\tag{LogReg}
\min_{x \in \mathbb{R}^n} \left( \frac{1}{m} \sum_{i=1}^m \ln(1 + \exp(-b_i \langle a_i, x \rangle)) + \frac{\lambda}{2} \|x\|^2 \right),
$$
where $\lambda \geq 0$ is the regularization coefficient (a model parameter).

Lets the LogReg problem be convex for $\lambda = 0$. What is the gradient of the objective function? Will it be strongly convex? What if you will add regularization with $\lambda > 0$?

\[
f(x) = \frac{1}{m} \sum_{i=1}^m \ln(1 + \exp(-b_i \langle a_i, x \rangle)) + \frac{\lambda}{2} \|x\|^2 
\]

$\forall x_1, x_2, \forall t$

\[
\frac{1}{m}\sum_{i=1}^{m}\ln(1 + \exp(-b_i(t\langle a_i, x_1\rangle + (1 - t)\langle a_i, x_2 \rangle ))) \leq
\]

\[
\leq \frac{1}{m}\sum_{i=1}^{m}t\ln(1 + \exp(-b_i\langle a_i, x_1 \rangle))  + (1-t)\ln (1 + \exp (-b_i \langle a_i, x_2 \rangle))
\]

\[
1 + \exp(-b_i (t \langle a_i, x_1 \rangle + (1 - t)\langle a_i, x_2 \rangle)) \leq
\]

\[
\leq (1 + \exp ( -b_i \langle a_i, x_1 \rangle ))^t (1 + \exp (-b_i \langle a_i, x_2 \rangle))^{1-t}
\]

\[
1 + \exp^t(-b_i \langle a_i, x_1 \rangle) \exp^{1-t}(-b_i \langle a_i, x_2 \rangle) \leq
\]

\[
\leq 1 + \exp^t(-b_i \langle a_i, x_1 \rangle) \exp^{1-t}(-b_i \langle a_i, x_2 \rangle) + ...
\]

Thus, we define that f(x) is convex by definition of convex functions. By the way, it's still convex with any $\lambda \geq 0$.

Now, we compute its gradient step by step.

Each individual term inside the summation in the loss function is:

\[
\ell_i(x) = \ln(1 + \exp(-b_i \langle a_i, x \rangle))
\]

Define:

\[
z_i = \langle a_i, x \rangle
\]

Then,

\[
\ell_i(x) = \ln(1 + \exp(-b_i z_i))
\]

Differentiate w.r.t. \( x \):

\[
\nabla_x \ell_i(x) = \frac{-b_i \exp(-b_i z_i)}{1 + \exp(-b_i z_i)} a_i
\]

Using the sigmoid function definition:

\[
\sigma(y) = \frac{1}{1 + \exp(-y)}
\]

We rewrite:

\[
\nabla_x \ell_i(x) = -b_i (1 - \sigma(b_i z_i)) a_i
\]

So the gradient of the summation term is:

\[
\nabla_x \left( \frac{1}{m} \sum_{i=1}^{m} \ln(1 + \exp(-b_i \langle a_i, x \rangle)) \right) = \frac{1}{m} \sum_{i=1}^{m} -b_i (1 - \sigma(b_i \langle a_i, x \rangle)) a_i
\]

The second term in the function is:

\[
\frac{\lambda}{2} \|x\|^2
\]

Since the gradient of \(\frac{1}{2} \|x\|^2\) is simply \( x \), we get:

\[
\nabla_x \left( \frac{\lambda}{2} \|x\|^2 \right) = \lambda x
\]

Thus, the full gradient is:

\[
\nabla f(x) = \frac{1}{m} \sum_{i=1}^{m} -b_i (1 - \sigma(b_i \langle a_i, x \rangle)) a_i + \lambda x
\]

Or more compactly:

\[
\nabla f(x) = -\frac{1}{m} \sum_{i=1}^{m} b_i (1 - \sigma(b_i \langle a_i, x \rangle)) a_i + \lambda x
\]

For the regularized logistic regression problem:

$$\min_{x \in \mathbb{R}^n} \left( \frac{1}{m} \sum_{i=1}^{m} \ln \left( 1 + \exp(-b_i \langle a_i, x \rangle) \right) + \frac{\lambda}{2} \|x\|^2 \right)$$

I'll determine the smoothness parameter $L$ and the strong convexity parameter $\mu$.

The smoothness parameter $L$ is the upper bound on the eigenvalues of the Hessian matrix of the objective function.

Let's compute the Hessian:

1. First, let's denote $f(x) = \frac{1}{m} \sum_{i=1}^{m} \ln(1 + \exp(-b_i \langle a_i, x \rangle)) + \frac{\lambda}{2}\|x\|^2$

2. The Hessian is:
$$\nabla^2 f(x) = \frac{1}{m} \sum_{i=1}^{m} \frac{\exp(-b_i \langle a_i, x \rangle)}{(1 + \exp(-b_i \langle a_i, x \rangle))^2} \cdot a_i a_i^T + \lambda I$$

3. We can simplify the first term using the logistic function $\sigma(z) = \frac{1}{1+e^{-z}}$:
$$\nabla^2 f(x) = \frac{1}{m} \sum_{i=1}^{m} \sigma(b_i \langle a_i, x \rangle) \cdot (1-\sigma(b_i \langle a_i, x \rangle)) \cdot a_i a_i^T + \lambda I$$

4. Note that $\sigma(z)(1-\sigma(z)) \leq \frac{1}{4}$ for all $z$ (maximum at $z=0$)

5. Therefore:
$$\nabla^2 f(x) \preceq \frac{1}{4m} \sum_{i=1}^{m} a_i a_i^T + \lambda I$$

6. The maximum eigenvalue of the Hessian is bounded by:
$$L = \frac{1}{4m} \lambda_{max}\left(\sum_{i=1}^{m} a_i a_i^T\right) + \lambda$$

If we define $A = [a_1, a_2, \ldots, a_m]^T$, then:
$$L = \frac{1}{4m} \lambda_{max}(A^T A) + \lambda = \frac{\|A\|_2^2}{4m} + \lambda$$

where $\|A\|_2$ is the spectral norm of matrix $A$.

The strong convexity parameter $\mu$ is the lower bound on the eigenvalues of the Hessian matrix.

1. From the Hessian expression:
$$\nabla^2 f(x) = \frac{1}{m} \sum_{i=1}^{m} \sigma(b_i \langle a_i, x \rangle)(1-\sigma(b_i \langle a_i, x \rangle)) \cdot a_i a_i^T + \lambda I$$

2. Since $\sigma(z)(1-\sigma(z)) \geq 0$ for all $z$, we have:
$$\nabla^2 f(x) \succeq \lambda I$$

3. Therefore, the strong convexity parameter is:
$$\mu = \lambda$$

For the regularized logistic regression problem:

- **Strong smoothness parameter**: $L = \frac{\|A\|_2^2}{4m} + \lambda$
- **Strong convexity parameter**: $\mu = \lambda$

The condition number of this optimization problem is therefore:
$$\kappa = \frac{L}{\mu} = \frac{\|A\|_2^2}{4m\lambda} + 1$$

This analysis shows that the regularization parameter $\lambda$ directly influences both the smoothness and convexity of the problem, and larger values of $\lambda$ improve the condition number of the problem.

**Conclusion**

I make realisation and experiments for heavy ball and Nesterovs methods. As a convergence criteria was chosen tol=10e-6 L2 norm weights not changing criteria. All methods were works. The best betta for heavy ball $\beta = 0.6$, for Nesterovs method $\beta = -1$, strategies, used changing from iteration to iteration betta works worst, than constant methods. For all methods I plotted graphics, showing method performance on the given dataset.
