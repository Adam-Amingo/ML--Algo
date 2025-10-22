# ML--Algo

## MY Machine Learning Journey 

``` Started my ML journey on the 1st of october . 
Have learnt a lot . Will start to give update on what i learn each day .
```
# 🧮 Logistic Regression — A Probabilistic Classifier

Logistic Regression is a **statistical model** used to estimate the **probability** of an event belonging to one of two categories (binary classification).

---

## 1️⃣ Linear Equation

The model starts by calculating a **linear combination** of inputs, similar to linear regression:

![z](https://latex.codecogs.com/svg.image?\bg_white%20z=\mathbf{w}\cdot\mathbf{x}+b)

> 💡 Here:
> - **w** → weight vector  
> - **x** → feature vector  
> - **b** → bias term  
> - **z** → raw model output (can range from −∞ to +∞)

---

## 2️⃣ The Sigmoid (Logistic) Function

To map the linear output `z` into a probability between 0 and 1, we pass it through the **Sigmoid function**:

![sigmoid](https://latex.codecogs.com/svg.image?\bg_white%20\sigma(z)=\frac{1}{1+e^{-z}})

> - If the output ≥ 0.5 → predict **class 1**
> - Otherwise → predict **class 0**

This is what makes Logistic Regression **probabilistic** — it gives a *likelihood* rather than a direct class.

---

## 3️⃣ Machine Learning as Optimization

Training a machine learning model is an **optimization problem**.

We want to find the parameters (**w**, **b**) that **minimize the error** between predicted probabilities and actual labels.

This is done through **Gradient Descent**, which iteratively updates weights in the opposite direction of the gradient of the loss function.

---

## 4️⃣ Loss Function — Binary Cross-Entropy (Log Loss)

Because Logistic Regression predicts probabilities, it cannot use Mean Squared Error (MSE).  
Instead, it uses the **Binary Cross-Entropy Loss**, also called **Log Loss**.

### 💡 Concept

Cross-Entropy heavily penalizes confident wrong predictions —  
e.g., predicting 0.99 when the true label is 0 gives a large penalty.

### 📘 Formula

For `m` training examples:

![J(w)](https://latex.codecogs.com/svg.image?\bg_white%20J(\mathbf{w})=-\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(h_{\mathbf{w}}(\mathbf{x}^{(i)}))+(1-y^{(i)})\log(1-h_{\mathbf{w}}(\mathbf{x}^{(i)}))\right])

where:

- ![y](https://latex.codecogs.com/svg.image?\bg_white%20y^{(i)}) → true label (0 or 1)  
- ![h](https://latex.codecogs.com/svg.image?\bg_white%20h_{\mathbf{w}}(\mathbf{x}^{(i)})) → predicted probability from the sigmoid

---

## 5️⃣ Gradient Descent Updates

Weights are updated using the **gradient of the loss**:

![gradient](https://latex.codecogs.com/svg.image?\bg_white%20\mathbf{w}:=\mathbf{w}-\alpha\frac{\partial{J(\mathbf{w})}}{\partial{\mathbf{w}}})

> - **α** → learning rate  
> - **∂J/∂w** → derivative of the loss function w.r.t. weights  

This process repeats until the loss converges to its minimum.

---

### 🧠 Summary

| Concept       | Equation                                                                                                                                                                                                         | Meaning                     |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Linear output | ![z](https://latex.codecogs.com/svg.image?\bg_white%20z=\mathbf{w}\cdot\mathbf{x}+b)                                                                                                                             | Weighted sum of inputs      |
| Sigmoid       | ![sigmoid](https://latex.codecogs.com/svg.image?\bg_white%20\sigma(z)=\frac{1}{1+e^{-z}})                                                                                                                        | Converts to probability     |
| Loss          | ![loss](https://latex.codecogs.com/svg.image?\bg_white%20J(\mathbf{w})=-\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(h_{\mathbf{w}}(\mathbf{x}^{(i)}))+(1-y^{(i)})\log(1-h_{\mathbf{w}}(\mathbf{x}^{(i)}))\right]) | Penalizes wrong predictions |
| Update rule   | ![gradient](https://latex.codecogs.com/svg.image?\bg_white%20\mathbf{w}:=\mathbf{w}-\alpha\frac{\partial{J(\mathbf{w})}}{\partial{\mathbf{w}}})                                                                  | Weight optimization         |

---

✨ *Logistic Regression bridges the gap between Linear Models and Probabilistic Classification — forming the foundation for Neural Networks and Deep Learning.*


