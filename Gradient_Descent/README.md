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


```


```

# ⚙️ Gradient Descent — Core Optimization Algorithm

Gradient Descent is an **iterative optimization algorithm** used to minimize a cost (loss) function by updating parameters in the opposite direction of the gradient.

---

## 🧮 1️⃣ Batch Gradient Descent (Full Gradient Descent)

Batch Gradient Descent computes the **gradient of the entire training dataset** before each update.

### 📘 Cost Function
The cost function for linear regression is:

![J(theta)](https://latex.codecogs.com/svg.image?&space;J(%5Ctheta)%3D%5Cfrac%7B1%7D%7B2m%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29%5E2)

### 📘 Parameter Update Rule
In Batch Gradient Descent, the parameters are updated as:

![theta update](https://latex.codecogs.com/svg.image?&space;%5Ctheta_j%3A%3D%5Ctheta_j-%5Calpha%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29x_j%5E%7B%28i%29%7D)

> Where:
> - ![theta](https://latex.codecogs.com/svg.image?&space;%5Ctheta_j) → model parameter  
> - ![alpha](https://latex.codecogs.com/svg.image?&space;%5Calpha) → learning rate  
> - ![h(x)](https://latex.codecogs.com/svg.image?&space;h_%7B%5Ctheta%7D%28x%29%3D%5Ctheta%5ETx) → hypothesis function  
> - ![m](https://latex.codecogs.com/svg.image?&space;m) → number of training examples  

### ✅ Advantages
- Converges **smoothly** and **stably** toward the global minimum (for convex functions).
- Provides **accurate** updates using all data points.

### ⚠️ Disadvantages
- **Computationally expensive** for large datasets.
- Requires the **entire dataset** to fit in memory.
- **Slow convergence** when data is massive.

---

## ⚙️ 2️⃣ Stochastic Gradient Descent (SGD)

Instead of computing the gradient over all `m` samples, SGD updates parameters **after each training example**.

### 📘 Update Rule

![SGD](https://latex.codecogs.com/svg.image?&space;%5Ctheta_j%3A%3D%5Ctheta_j-%5Calpha%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29x_j%5E%7B%28i%29%7D)

### ✅ Advantages
- **Much faster** for large datasets.
- Enables **online learning** — model updates continuously as new data arrives.
- Can **escape local minima** due to its noisy updates.

### ⚠️ Disadvantages
- **Noisy convergence** — loss fluctuates heavily before stabilizing.
- May not reach the exact global minimum but oscillate around it.

---

## 📊 3️⃣ Comparison Summary

| Type                    | Update Frequency  | Convergence     | Computation | Memory | Best For             |
| ----------------------- | ----------------- | --------------- | ----------- | ------ | -------------------- |
| **Batch GD**            | After all samples | Smooth & stable | Expensive   | High   | Small datasets       |
| **Stochastic GD (SGD)** | After each sample | Noisy but fast  | Cheap       | Low    | Large/streaming data |

---

### ✨ Intuition

Batch Gradient Descent takes **large but steady steps** by averaging gradients from the whole dataset.  
Stochastic Gradient Descent takes **many quick steps**, bouncing around the valley but often finding a good minimum faster.

---

🧠 *Both methods form the backbone of modern optimization in machine learning — from linear regression to deep neural networks.*
