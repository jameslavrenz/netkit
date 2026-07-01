# Multilayer Perceptrons (MLPs) and Backpropagation — A Complete Engineering Explanation

> **Note:** This document explains neural network *training* theory (backpropagation, loss functions, gradient descent). **netkit is inference-only** — it runs forward passes on pre-trained float32 weights and does not implement training. For netkit usage, start with [GETTING_STARTED.md](GETTING_STARTED.md) and [MODEL_FORMAT.md](MODEL_FORMAT.md).

---

## 1. What a Neural Network Actually Is (Engineering View)

A neural network, at its core, is just a **composable function approximator**.

From an engineering perspective:

- You are building a system that maps inputs → outputs
- That system is composed of repeated simple computational blocks
- Each block does:
  - a linear transformation (like a weighted sum / FIR filter)
  - followed by a nonlinear function (like a saturation or lookup table)

So an MLP is nothing more than:

> A stack of parameterized nonlinear functions trained to approximate a mapping from $\mathbb{R}^n \to \mathbb{R}^m$

---

## 2. The Basic Building Block (Neuron)

A single neuron computes:

$$
z = w^T x + b
$$

$$
a = \sigma(z)
$$

Where:
- $x \in \mathbb{R}^n$: input vector
- $w \in \mathbb{R}^n$: weights (learned parameters)
- $b \in \mathbb{R}$: bias (learned offset)
- $z$: linear combination
- $a$: output activation
- $\sigma(\cdot)$: nonlinear activation function

### Engineering intuition

Think of this like:

- A weighted sum → like a DSP dot product (FIR filter tap)
- Bias → DC offset
- Activation → nonlinear saturation or transfer curve

Without the activation function, stacking layers collapses into a single linear transform. That would be equivalent to:

> One giant matrix multiply → no expressive power

---

## 3. Activation Functions

Activation functions introduce **nonlinearity**, which is what makes neural networks useful.

### 3.1 Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Properties:
- Output range: (0, 1)
- Historically used for probabilities

Engineering issue:
- Saturates for large |x|
- Causes vanishing gradients

---

### 3.2 Tanh

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

Properties:
- Output range: (-1, 1)
- Zero-centered (better than sigmoid)

Still suffers from saturation.

---

### 3.3 ReLU (Rectified Linear Unit)

$$
\text{ReLU}(x) = \max(0, x)
$$

Properties:
- Simple
- Efficient (important in embedded systems)
- Sparse activations

Engineering analogy:
- Like a diode: blocks negative values, passes positive

Why it dominates:
- Avoids saturation in positive region
- Cheap computation (no exponentials)

---

### 3.4 Why Nonlinearity Matters

Without nonlinearities:

$$
f(x) = W_3(W_2(W_1 x)) = W x
$$

All layers collapse into one matrix.

So depth becomes useless.

Nonlinearity is what turns the system into a **universal function approximator**.

---

## 4. Multilayer Perceptron (MLP) Structure

An MLP is a sequence of layers:

$$
x \rightarrow \text{Layer}_1 \rightarrow \text{Layer}_2 \rightarrow ... \rightarrow \text{Layer}_L \rightarrow \hat{y}
$$

Each layer:

$$
a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})
$$

Where:
- $a^{(0)} = x$
- $a^{(L)} = \hat{y}$

---

## 5. Forward Pass (What Happens at Runtime)

The forward pass is just evaluation of the function.

For each layer:

### Step 1: Linear transform

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

### Step 2: Nonlinear activation

$$
a^{(l)} = \sigma(z^{(l)})
$$

---

### Engineering interpretation

This is equivalent to:

- Matrix multiplication pipeline
- Followed by elementwise nonlinear transform
- Repeated N times

If you were implementing this on embedded hardware:

- This is GEMM + vector activation kernel
- Highly parallelizable
- Cache/memory bound in practice

---

## 6. Loss Function (What “Error” Means)

The loss function defines how wrong the network is.

### 6.1 Mean Squared Error (Regression)

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

Interpretation:
- Penalizes squared deviation
- Equivalent to L2 energy of error signal

---

### 6.2 Cross Entropy (Classification)

For binary classification:

$$
\mathcal{L} = - \left(y \log \hat{y} + (1 - y)\log(1 - \hat{y}) \right)
$$

Engineering intuition:
- Measures divergence between probability distributions
- Strong penalty for confident wrong predictions

---

## 7. Why We Need Backpropagation

We want to adjust weights to minimize loss:

$$
\min_{W,b} \mathcal{L}
$$

But:

- The function is nested
- Many layers depend on each other
- Direct differentiation is expensive without structure

So we use:

> Chain rule + dynamic programming = backpropagation

---

## 8. Backpropagation (Core Idea)

Backprop computes gradients efficiently:

Instead of recomputing partial derivatives repeatedly, we reuse intermediate results.

---

## 9. The Chain Rule (Foundation)

If:

$$
y = f(g(x))
$$

Then:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

For deep networks:

- You apply this repeatedly across layers

---

## 10. Backprop in a Single Layer

Recall:

$$
z = W a + b
$$

$$
a = \sigma(z)
$$

Loss:

$$
\mathcal{L}(a)
$$

We want:

$$
\frac{\partial \mathcal{L}}{\partial W}
$$

---

## 11. Step-by-Step Gradient Flow

### 11.1 Output error signal

Define:

$$
\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}}
$$

This is the key object in backprop.

It represents:

> “How much this layer contributed to the final error”

---

### 11.2 Gradient for weights

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
$$

Interpretation:
- Outer product of error signal and input activation
- Very similar to correlation in signal processing

---

### 11.3 Gradient for bias

$$
\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}
$$

---

### 11.4 Propagating error backward

$$
\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})
$$

Where:
- $\odot$ = elementwise product
- $\sigma'(z)$ = derivative of activation

---

## 12. Activation Derivatives

### Sigmoid:

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

### Tanh:

$$
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)
$$

### ReLU:

$$
\text{ReLU}'(x) =
\begin{cases}
1 & x > 0 \\
0 & x \le 0
\end{cases}
$$

Engineering note:
- ReLU derivative is basically a **mask operation**

---

## 13. Full Backprop Algorithm (MLP)

### Forward pass:
For each layer:

1. Compute:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

2. Apply activation:

$$
a^{(l)} = \sigma(z^{(l)})
$$

---

### Backward pass:

1. Compute output error:

$$
\delta^{(L)} = \nabla_a \mathcal{L} \odot \sigma'(z^{(L)})
$$

2. For each layer going backward:

$$
\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})
$$

3. Gradients:

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
$$

---

## 14. Gradient Descent (Weight Update Rule)

Once gradients are computed:

$$
W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}
$$

$$
b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
$$

Where:
- $\eta$ = learning rate

---

## 15. Engineering Interpretation of Training

Training is:

> A feedback control system

- Forward pass = system response
- Loss = error measurement
- Backprop = sensitivity analysis
- Gradient descent = control law update

This is very similar to:
- adaptive filters (LMS algorithm)
- Kalman filter tuning
- system identification loops

---

## 16. Why Deep Networks Work

Depth allows:

- hierarchical feature extraction
- composition of nonlinear transforms
- exponential reuse of intermediate representations

Mathematically:
- each layer composes functions
- composition increases expressiveness exponentially

---

## 17. Common Failure Modes

### 17.1 Vanishing gradients
- derivatives shrink through layers
- sigmoid/tanh worsen this

### 17.2 Exploding gradients
- derivatives grow uncontrollably

### 17.3 Saturation
- activations stuck near limits

---

## 18. Why ReLU + Modern Nets Work Better

ReLU helps because:

- derivative is constant (1 or 0)
- avoids saturation in positive region
- keeps gradient flow stable

---

## 19. Summary Mental Model

An MLP is:

1. A chain of affine transforms:
   
$$
Wx + b
$$

2. Interleaved with nonlinearities:

$$
\sigma(\cdot)
$$

3. Trained using:
   - chain rule
   - gradient descent
   - error propagation backward through graph

---

## 20. Final Engineering Analogy

If you're coming from embedded/DSP:

| Neural Net Concept | DSP Equivalent |
|-------------------|---------------|
| Layer | Filter stage |
| Weights | FIR coefficients |
| Activation | Nonlinear amplifier |
| Forward pass | Signal propagation |
| Backprop | Sensitivity / adjoint system |
| Loss | Error energy |
| Training | Adaptive filtering |

---

## End
