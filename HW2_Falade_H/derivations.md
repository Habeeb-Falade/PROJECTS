# Mathematical derivations

## Q1(a): composing two affine layers

With the identity activation, h = W1 x + b1, so

\[
z=W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2).
\]

Thus W_eff = W2 W1 has shape C × d and b_eff = W2 b1 + b2 has shape C.
The pairwise decision boundary for classes i and j is
(W_eff[i,:] − W_eff[j,:]) x + b_eff[i] − b_eff[j] = 0, a hyperplane.
Repeated affine composition is still affine; it cannot produce nonlinear
classification boundaries. Softmax changes the probabilities but preserves
the logit ordering and therefore does not change these argmax boundaries.
A narrow hidden layer can additionally constrain the rank of W_eff.

## Q1(b): cross-entropy and chain rule

Starting from ell = −z_y + log(sum_j exp(z_j)), differentiate component k:

\[
\frac{\partial\ell}{\partial z_k}
=-\mathbf1\{k=y\}+\frac{e^{z_k}}{\sum_j e^{z_j}}
=p_k-\mathbf1\{k=y\}.
\]

Let delta_z = p − e_y (shape C), r = 1{a > 0} (shape m), and
delta_a = (W2^T delta_z) elementwise-multiplied by r (shape m).
The mask is zero at a = 0, matching the specified ReLU derivative.
Because dz = dW2 h + W2 dh + db2 and da = dW1 x + db1,
the gradients are

\[
\nabla_{W_2}\ell=\delta_z h^T\quad(C\times m),\qquad
\nabla_{b_2}\ell=\delta_z\quad(C),
\]
\[
\nabla_h\ell=W_2^T\delta_z\quad(m),\qquad
\nabla_a\ell=\delta_a\quad(m),
\]
\[
\nabla_{W_1}\ell=\delta_a x^T\quad(m\times d),\qquad
\nabla_{b_1}\ell=\delta_a\quad(m).
\]

For the specified example,
a = (1, −1)^T, h = (1, 0)^T, and z = (1, −1)^T.
Write q = e²/(e²+1) = 0.880797077977882 approximately.
Then p = (q, 1−q)^T = (0.880797077978, 0.119202922022)^T,
ell = log(1+e²) = 2.126928011043, and delta_z = (q, −q)^T.
The ReLU mask is (1,0)^T, W2^T delta_z = (2q, −2q)^T,
and delta_a = (2q,0)^T. Therefore

\[
\nabla_{W_2}\ell=\begin{bmatrix}q&0\\-q&0\end{bmatrix},\quad
\nabla_{b_2}\ell=\begin{bmatrix}q\\-q\end{bmatrix},\quad
\nabla_{W_1}\ell=\begin{bmatrix}2q&-4q\\0&0\end{bmatrix},\quad
\nabla_{b_1}\ell=\begin{bmatrix}2q\\0\end{bmatrix}.
\]

Numerically, the nonzero first-layer entries are 2q = 1.761594155956
and −4q = −3.523188311912. All four gradient matrices/vectors use
the column-vector convention; the implementation stores vectors as rank-1
arrays, which gives the same values and outer products.

## Q1(c): independent numerical check

`verify_gradients.py` computes the forward pass and chain-rule gradients
with NumPy float64, then separately constructs float64 PyTorch leaf
parameters and calls cross_entropy(...).backward(). It compares every
entry of all four parameter gradients. The measured largest absolute
difference is **0.0** across all four gradients with Python 3.12.14,
PyTorch 2.6.0+cpu, and NumPy 2.3.5. The complete values are saved in
`results/gradient_results.json`; no training or finite-difference
approximation is involved.

## Q2(b): MLP parameters

The input-to-hidden layer has 784H weights and H biases, and the
hidden-to-output layer has 10H weights and 10 biases. Thus

\[
P(H)=784H+H+10H+10=795H+10.
\]

P(32) = 25,450; P(128) = 101,770. The verification script independently
checks both against sum(p.numel() for p in model.parameters()).

## Q3(a): spatial dimensions and convolution parameters

For a convolution or pool with input size n, kernel k, padding p,
dilation d, and stride s, the output size per spatial dimension is

\[
n_{out}=\left\lfloor\frac{n+2p-d(k-1)-1}{s}+1\right\rfloor.
\]

All convolutions here have dilation 1 and stride 1. ReLU preserves shape.

| Stage | Calculation per spatial dimension | Output shape |
|---|---|---|
| Input | — | B × 1 × 28 × 28 |
| Conv1 | (28 − 5)/1 + 1 = 24 | B × 6 × 24 × 24 |
| ReLU1 | unchanged | B × 6 × 24 × 24 |
| Pool1 | floor((24 − 2)/2 + 1) = 12 | B × 6 × 12 × 12 |
| Conv2 | (12 + 2 − 3)/1 + 1 = 12 | B × 12 × 12 × 12 |
| ReLU2 | unchanged | B × 12 × 12 × 12 |
| Pool2 | floor((12 − 2)/2 + 1) = 6 | B × 12 × 6 × 6 |
| Flatten | 12 × 6 × 6 = 432 | B × 432 |
| Classifier | 432 → 10 | B × 10 |

Conv1: 6(1 × 5 × 5 + 1) = 156 parameters.
Conv2: 12(6 × 3 × 3 + 1) = 660 parameters.
Classifier: 10(432 + 1) = 4,330 parameters.
Total: 156 + 660 + 4,330 = **5,146** parameters.

Each output-channel filter and its bias are reused at every spatial
position. Positions produce distinct activations, not distinct learned
weights; multiplying the filter count by output height and width would
therefore overcount the parameters. The dummy-batch verification uses
B = 2 and checks every layer shape and all three learned-layer counts.
