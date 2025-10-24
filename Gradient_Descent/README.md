# Gradient Descent and Stochastic Gradient Descent (SGD)

## 1️⃣ Overview

Gradient Descent is an optimization algorithm used to minimize a **cost function** by iteratively moving in the direction of the steepest descent as defined by the negative of the gradient. It’s a cornerstone method for training **machine learning models** such as linear regression and logistic regression.

---

## 2️⃣ The General Idea

The goal is to minimize the cost function:

![J(θ)](https://latex.codecogs.com/svg.image?J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^{2})

where:
- \( m \) — number of training examples  
- \( h_{\theta}(x^{(i)}) \) — predicted value  
- \( y^{(i)} \) — actual value  

The parameter update rule is:

![theta update](https://latex.codecogs.com/svg.image?\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta))

where:
- \( \alpha \) — learning rate  
- \( \frac{\partial}{\partial\theta_j}J(\theta) \) — gradient of the cost function with respect to parameter \( \theta_j \)

---

## 3️⃣ Batch Gradient Descent (BGD)

### Equation

In **Batch Gradient Descent**, the update for each parameter uses **all training examples**:

![BGD equation](https://latex.codecogs.com/svg.image?\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)})

### When to Use
- When the dataset is **small to medium-sized** and can fit in memory.
- When **accurate gradient estimates** are needed.

### Advantages
✅ Converges smoothly to the global minimum for convex functions.  
✅ More stable updates — less noise in the learning process.

### Disadvantages
❌ Slow for large datasets — every update requires a full pass through all data.  
❌ Computationally expensive.

---

## 4️⃣ Stochastic Gradient Descent (SGD)

### Equation

In **Stochastic Gradient Descent**, the update happens **after every training example**:

![SGD equation](https://latex.codecogs.com/svg.image?\theta_j:=\theta_j-\alpha(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)})

### When to Use
- When working with **very large datasets** or **streaming data**.
- When quick convergence or **online learning** is needed.

### Advantages
✅ Much faster for large datasets.  
✅ Can escape local minima due to noisy updates.  
✅ Suitable for real-time and online learning.

### Disadvantages
❌ The loss function fluctuates (noisy convergence).  
❌ May overshoot the minimum if the learning rate is not well-tuned.

---

## 5️⃣ Comparison Table

| Feature              | Batch Gradient Descent | Stochastic Gradient Descent    |
| -------------------- | ---------------------- | ------------------------------ |
| Update per iteration | Uses all samples       | Uses one sample                |
| Speed                | Slow on large data     | Fast, efficient                |
| Stability            | Stable convergence     | Noisy convergence              |
| Memory use           | High                   | Low                            |
| Ideal for            | Small datasets         | Large datasets, streaming data |

---

## 6️⃣ Summary Visualization

Gradient Descent can be visualized as moving downhill on a cost surface — the learning rate \( \alpha \) determines the size of each step:

![GD surface](https://latex.codecogs.com/svg.image?\theta_{new}=\theta_{old}-\alpha\nabla_{\theta}J(\theta))

---

## 🧠 Key Takeaway

- **Batch Gradient Descent** → precise but slow.  
- **Stochastic Gradient Descent** → fast but noisy.  
- **Mini-batch Gradient Descent** (not covered here) offers a balance between both.

---
