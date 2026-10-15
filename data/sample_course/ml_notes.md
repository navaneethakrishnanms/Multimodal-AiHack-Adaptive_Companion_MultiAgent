# Introduction to Machine Learning — Course Notes
## Chapter 1: What is Machine Learning?

Machine learning is the study of algorithms that improve their performance on a task
through experience. In the classical formulation, a program learns from experience E
with respect to a task T and a performance measure P if its performance at T, as
measured by P, improves with experience E.

There are three main paradigms:

**Supervised learning** uses labelled examples. The training set consists of input-output
pairs, and the goal is to learn a mapping from inputs to outputs that generalizes to
unseen data. Examples: spam classification, house price prediction, medical diagnosis.

**Unsupervised learning** works with unlabelled data. The goal is to discover structure:
clusters, low-dimensional representations, or anomalous points. Examples: customer
segmentation, dimensionality reduction, density estimation.

**Reinforcement learning** trains an agent that interacts with an environment. The agent
observes states, takes actions, and receives rewards. The goal is to learn a policy that
maximizes expected cumulative reward. Examples: game playing, robot control, ad placement.

## Chapter 2: Linear Regression

Linear regression models the relationship between a continuous target y and one or more
input features x. The model makes the prediction:

    y_hat = w · x + b

where w is a weight vector and b is the bias term. Training means choosing w and b to
minimize a loss function over the training data.

The most common loss is the mean squared error (MSE):

    MSE = (1/n) * sum_i (y_i - y_hat_i)^2

For linear regression with MSE, the optimal parameters have a closed-form solution called
the normal equations:

    w = (X^T X)^{-1} X^T y

In practice, when the number of features is large, gradient descent is preferred because
inverting X^T X costs O(d^3) time and can be numerically unstable.

## Chapter 3: Gradient Descent

Gradient descent is an iterative optimization algorithm. Starting from an initial guess
for the parameters, it repeatedly takes steps in the direction of the negative gradient
of the loss:

    theta := theta - alpha * grad_theta L(theta)

Here alpha is the learning rate, the most important hyperparameter.

- If the learning rate is too small, convergence is slow and training may appear to stall.
- If the learning rate is too large, the algorithm can overshoot the minimum and diverge.
- A good learning rate makes the loss decrease smoothly and steadily.

The gradient of the MSE loss for linear regression is:

    grad_w L = (2/n) * X^T (X w - y)

Stochastic gradient descent (SGD) approximates the gradient using a small random subset
of examples (a mini-batch) at each step. This makes each step much cheaper and often
speeds up convergence in practice, at the cost of a noisier loss curve.

## Chapter 4: Overfitting and Regularization

A model overfits when it fits the training data very well but generalizes poorly to new
data. Overfitting is most likely when the model is complex relative to the amount of
training data.

The standard remedy is regularization, which adds a penalty on model complexity to the
loss. Two common forms:

- **L2 regularization (ridge)**: add lambda * ||w||^2 to the loss. This shrinks weights
  smoothly toward zero and is differentiable everywhere.
- **L1 regularization (lasso)**: add lambda * ||w||_1 to the loss. This encourages sparse
  solutions where some weights become exactly zero, performing feature selection.

The regularization strength lambda is tuned on a validation set, never on the test set.

Train/validation/test splits: the training set fits parameters, the validation set tunes
hyperparameters, and the test set gives an unbiased final estimate of generalization.
