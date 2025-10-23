# 📈 Linear Regression — Predicting Continuous Outcomes

Linear Regression is a **supervised learning algorithm** used to model the **relationship between input variables (features)** and a **continuous output variable (target)**.  
It assumes a **linear relationship** between the inputs and the target.

---

## 1️⃣ The Hypothesis Function

In **matrix form**, the model predicts the output vector **ŷ** as:

![h](https://latex.codecogs.com/svg.image?\bg_white\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}+\mathbf{b})

> 💡 Where:
> - **X** → matrix of input features (size: *m × n*)  
> - **w** → column matrix of weights (size: *n × 1*)  
> - **b** → bias term (scalar or broadcasted)  
> - **ŷ** → predicted outputs (size: *m × 1*)  

---

## 2️⃣ Goal of Linear Regression

We want to find parameters **W** and **b** such that the predictions `ŷ` are **as close as possible** to the true outputs **y**.  

This means minimizing the **difference (error)** between predicted and actual values.

---

## 3️⃣ Cost Function — Mean Squared Error (MSE)

In matrix notation, the **Mean Squared Error** (MSE) is given by:

![J](https://latex.codecogs.com/svg.image?\bg_white J(\mathbf{w},b)=\frac{1}{2m}(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y})^T(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y}))

> 💡 Here:
> - **m** → number of training examples  
> - **y** → actual output matrix (*m × 1*)  
> - **ŷ = Xw + b** → predicted output matrix (*m × 1*)  

The factor `1/2m` simplifies derivative computation.

---

## 4️⃣ Gradient Descent — Learning the Best Parameters

To minimize the cost, we use **Gradient Descent**, which updates the parameters iteratively:

![update](https://latex.codecogs.com/svg.image?\bg_white\mathbf{w}:=\mathbf{w}-\alpha\frac{1}{m}\mathbf{X}^T(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y}))

![bupdate](https://latex.codecogs.com/svg.image?\bg_white b:=b-\alpha\frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)}))

> - **α** → learning rate  
> - Updates continue until convergence (minimum cost)

---

## 5️⃣ Ordinary Least Squares (OLS) — The Analytical Solution

Instead of iterative updates, **OLS** gives a **closed-form solution** for the optimal weights that minimize the cost function.

### 📘 Formula

![ols](https://latex.codecogs.com/svg.image?\bg_white\mathbf{w}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y})

> 💡 Notes:
> - **X** → matrix of input features  
> - **y** → column matrix of target values  
> - **w** → matrix of optimal weights  

This solution directly minimizes the **sum of squared errors**, giving the **best-fit regression plane (or line)**.

---

## 6️⃣ The Regression Line (Simple Case)

For a single feature (simple linear regression), the hypothesis simplifies to:

![line](https://latex.codecogs.com/svg.image?\bg_white\hat{y}=w_1x+b)

> - `w₁` → slope of the line  
> - `b` → intercept (value of y when x = 0)

---

### 🧠 Summary

| Concept          | Equation                                                                                                                                                                    | Meaning                              |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Hypothesis       | ![h](https://latex.codecogs.com/svg.image?\bg_white\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}+\mathbf{b})                                                                        | Predicts output vector               |
| Cost Function    | ![J](https://latex.codecogs.com/svg.image?\bg_white J(\mathbf{w},b)=\frac{1}{2m}(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y})^T(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y})) | Measures prediction error            |
| Gradient Descent | ![update](https://latex.codecogs.com/svg.image?\bg_white\mathbf{w}:=\mathbf{w}-\alpha\frac{1}{m}\mathbf{X}^T(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y}))                   | Iterative optimization               |
| OLS Solution     | ![ols](https://latex.codecogs.com/svg.image?\bg_white\mathbf{w}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y})                                                        | Analytical solution (direct weights) |

---

✨ *Linear Regression (in matrix form) is the cornerstone of statistical learning — simple, interpretable, and the foundation for more complex models like Ridge and Lasso Regression.*



```








```






# 📊 Multivariate Linear Regression — (4 Features, Matrix Form)

Linear Regression models the relationship between **multiple input features** and a **continuous output variable**.  
It assumes the output is a **linear combination** of all input features.

---

## 1️⃣ Model Representation

For a dataset with **4 features**, the hypothesis in **matrix form** is:

![h](https://latex.codecogs.com/svg.image?\bg_white\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}+\mathbf{b})

> 💡 Where:
> - **X** → Input matrix of size *(m × 4)*  
>   \[
>   \mathbf{X} = 
>   \begin{bmatrix}
>   x_{11} & x_{12} & x_{13} & x_{14} \\
>   x_{21} & x_{22} & x_{23} & x_{24} \\
>   \vdots & \vdots & \vdots & \vdots \\
>   x_{m1} & x_{m2} & x_{m3} & x_{m4}
>   \end{bmatrix}
>   \]
> - **w** → Column matrix of weights *(4 × 1)*  
>   \[
>   \mathbf{w}=
>   \begin{bmatrix}
>   w_1\\
>   w_2\\
>   w_3\\
>   w_4
>   \end{bmatrix}
>   \]
> - **b** → Bias term (scalar, added to all predictions)  
> - **ŷ** → Predicted outputs *(m × 1)*  

So each prediction is computed as:

![yhat](https://latex.codecogs.com/svg.image?\bg_white\hat{y}^{(i)}=w_1x_1^{(i)}+w_2x_2^{(i)}+w_3x_3^{(i)}+w_4x_4^{(i)}+b)

---

## 2️⃣ Cost Function — Mean Squared Error (MSE)

The cost function measures the average squared difference between the predicted and actual values:

![J](https://latex.codecogs.com/svg.image?\bg_white J(\mathbf{w},b)=\frac{1}{2m}(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y})^T(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y}))

> 💡 Intuition:
> - Penalizes large deviations between predictions and actual labels.
> - The goal is to minimize this cost.

---

## 3️⃣ Gradient Descent — Iterative Optimization

To minimize the cost function, we update parameters iteratively:

**Weight update rule:**
![update](https://latex.codecogs.com/svg.image?\bg_white\mathbf{w}:=\mathbf{w}-\alpha\frac{1}{m}\mathbf{X}^T(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y}))

**Bias update rule:**
![bupdate](https://latex.codecogs.com/svg.image?\bg_white b:=b-\alpha\frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)}))

> - **α** → learning rate  
> - Continue until the cost function converges to its minimum.

---

## 4️⃣ Ordinary Least Squares (OLS) — Closed Form Solution

OLS provides a **direct analytical solution** (no iteration) to find the optimal weights:

![ols](https://latex.codecogs.com/svg.image?\bg_white\mathbf{w}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y})

Once **w** is computed, you can find **b** as:

![b](https://latex.codecogs.com/svg.image?\bg_white b=\bar{y}-\bar{\mathbf{X}}\mathbf{w})

> 💡 Where:
> - **X** → input matrix (*m × 4*)  
> - **y** → target column (*m × 1*)  
> - **w** → learned weight column (*4 × 1*)  
> - **b** → scalar bias (intercept)

---

## 5️⃣ Example — Predicting House Prices 🏠

Let’s say we have 4 features:

| Feature | Description               |
| ------- | ------------------------- |
| x₁      | Number of rooms           |
| x₂      | Size (in square meters)   |
| x₃      | Distance from city center |
| x₄      | Age of house              |

The model becomes:

![example](https://latex.codecogs.com/svg.image?\bg_white\hat{y}=w_1x_1+w_2x_2+w_3x_3+w_4x_4+b)

The coefficients (**w₁–w₄**) represent how strongly each feature influences the predicted price.

---

## 🧠 Summary

| Concept          | Equation                                                                                                                                                                    | Meaning                           |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| Hypothesis       | ![h](https://latex.codecogs.com/svg.image?\bg_white\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}+\mathbf{b})                                                                        | Predicts output vector            |
| Cost Function    | ![J](https://latex.codecogs.com/svg.image?\bg_white J(\mathbf{w},b)=\frac{1}{2m}(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y})^T(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y})) | Measures prediction error         |
| Gradient Descent | ![update](https://latex.codecogs.com/svg.image?\bg_white\mathbf{w}:=\mathbf{w}-\alpha\frac{1}{m}\mathbf{X}^T(\mathbf{X}\mathbf{w}+\mathbf{b}-\mathbf{y}))                   | Iteratively finds minimum loss    |
| OLS Solution     | ![ols](https://latex.codecogs.com/svg.image?\bg_white\mathbf{w}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y})                                                        | Analytical least-squares solution |

---

✨ *Linear Regression with 4 features generalizes easily to any number of predictors — forming the mathematical foundation for Ridge, Lasso, and even Neural Network layers.*


# ⚖️ Ordinary Least Squares (OLS) vs Ridge Regression

Both **OLS** and **Ridge Regression** are linear models used to estimate coefficients \( \beta \) that best fit the relationship between features \( X \) and target \( y \).  
The main difference lies in how they handle **model complexity** and **overfitting**.

---

```


```

## 1️⃣ Ordinary Least Squares (OLS)

OLS minimizes the **sum of squared errors** between predicted and actual values.

### 📘 Objective Function

![ols_obj](https://latex.codecogs.com/svg.image?\bg_white J(\beta)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^{2})

or in **matrix form**:

![ols_matrix](https://latex.codecogs.com/svg.image?\bg_white J(\beta)=\frac{1}{2m}(X\beta-y)^{T}(X\beta-y))

### 🧮 Closed-Form Solution

OLS provides an analytical solution for the optimal parameters:

![ols_solution](https://latex.codecogs.com/svg.image?\bg_white\hat{\beta}_{OLS}=(X^{T}X)^{-1}X^{T}y)

> 💡 OLS assumes:
> - No multicollinearity among features  
> - Homoscedasticity (constant variance)  
> - Independence of errors  

---

## 2️⃣ Ridge Regression (L2 Regularization)

Ridge Regression extends OLS by adding a **penalty term** on the magnitude of coefficients.

### 📘 Objective Function

![ridge_obj](https://latex.codecogs.com/svg.image?\bg_whiteJ(\beta)=\frac{1}{2m}(X\beta-y)^{T}(X\beta-y)+\lambda\|\beta\|_{2}^{2})

> - \( \lambda \geq 0 \): regularization strength  
> - \( \|\beta\|_{2}^{2} = \sum_{j=1}^{n}\beta_{j}^{2} \): L2 norm  

### 🧮 Closed-Form Solution

The Ridge solution modifies OLS by adding \( \lambda I \) to stabilize inversion:

![ridge_solution](https://latex.codecogs.com/svg.image?\bg_white\hat{\beta}_{Ridge}=(X^{T}X+\lambdaI)^{-1}X^{T}y)

> 🔍 When \( \lambda = 0 \), Ridge Regression becomes identical to OLS.  
> As \( \lambda \) increases, coefficients shrink toward zero, reducing overfitting.

---

## 3️⃣ Comparison Summary

| Concept                 | OLS                                                                                                   | Ridge Regression                                                                                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Objective**           | ![ols_J](https://latex.codecogs.com/svg.image?\bg_whiteJ(\beta)=\frac{1}{2m}(X\beta-y)^{T}(X\beta-y)) | ![ridge_J](https://latex.codecogs.com/svg.image?\bg_whiteJ(\beta)=\frac{1}{2m}(X\beta-y)^{T}(X\beta-y)+\lambda\|\beta\|_{2}^{2}) |
| **Solution**            | ![ols_sol](https://latex.codecogs.com/svg.image?\bg_white\hat{\beta}_{OLS}=(X^{T}X)^{-1}X^{T}y)       | ![ridge_sol](https://latex.codecogs.com/svg.image?\bg_white\hat{\beta}_{Ridge}=(X^{T}X+\lambdaI)^{-1}X^{T}y)                     |
| **Regularization Term** | None                                                                                                  | \( \lambda\|\beta\|_{2}^{2} \)                                                                                                   |
| **Effect of λ**         | —                                                                                                     | Controls coefficient shrinkage                                                                                                   |
| **Overfitting**         | Prone when multicollinearity exists                                                                   | Reduces overfitting and variance                                                                                                 |
| **Interpretability**    | Easier                                                                                                | Slightly less intuitive (coefficients biased toward 0)                                                                           |

---

## 🧠 Key Takeaways

- **OLS** finds parameters that minimize residual error — perfect when features are independent.  
- **Ridge Regression** adds an L2 penalty to improve generalization and numerical stability.  
- Ridge helps when \( X^T X \) is nearly singular or features are highly correlated.

---

✨ *Ridge Regression = OLS + Penalty Term → More robust, less variance, smoother generalization.*

