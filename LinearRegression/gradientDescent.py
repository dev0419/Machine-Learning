import numpy as np
import matplotlib.pyplot as plt

# Sample data
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([2.0, 4.0, 5.0, 4.0, 5.0])

# Compute cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    f_wb = w * x + b
    cost = np.sum((f_wb - y) ** 2) / (2 * m)
    return cost

# Compute gradient
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    f_wb = w * x + b
    dj_dw = np.sum((f_wb - y) * x) / m
    dj_db = np.sum(f_wb - y) / m
    return dj_dw, dj_db

# Gradient descent
def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    w = w_init
    b = b_init
    J_history = []
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = compute_cost(x, y, w, b)
        J_history.append(cost)
    
    return w, b, J_history

# Initialize parameters
w_init = 0
b_init = 0
iterations = 10000
learning_rate = 0.01

# Run gradient descent
w_final, b_final, J_hist = gradient_descent(x_train, y_train, w_init, b_init, learning_rate, iterations)
print(f"(w, b) found by gradient descent: ({w_final:.4f}, {b_final:.4f})")

# Create a meshgrid for contour plot
w_range = np.linspace(-2, 2, 100)
b_range = np.linspace(-2, 2, 100)
W, B = np.meshgrid(w_range, b_range)
Z = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = compute_cost(x_train, y_train, W[i, j], B[i, j])

# Plot contour plot
plt.contourf(W, B, Z, levels=20, cmap="viridis")
plt.colorbar(label="Cost")
plt.scatter(w_final, b_final, color='red', marker='x', label='Final Parameters')
plt.xlabel("Parameter w")
plt.ylabel("Parameter b")
plt.title("Contour Plot of Cost Function with Gradient Descent")
plt.legend()
plt.show()
