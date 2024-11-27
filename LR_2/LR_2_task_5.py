import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Параметри варіанту
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X**2 + X + 2 + np.random.randn(m, 1)

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)

# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Графік
plt.figure(figsize=(10, 6))

# Дані та лінійна регресія
plt.scatter(X, y, color="blue", label="Дані")
plt.plot(X, y_pred_linear, color="red", label="Лінійна регресія")

# Поліноміальна регресія
X_sorted = np.sort(X, axis=0)  # Сортування для плавності кривої
y_sorted_poly = poly_reg.predict(poly_features.transform(X_sorted))
plt.plot(X_sorted, y_sorted_poly, color="green", label="Поліноміальна регресія")

# Декорування графіку
plt.title("Порівняння лінійної та поліноміальної регресії")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# Виведення коефіцієнтів
print("Лінійна регресія:")
print(f"Коефіцієнти: {lin_reg.coef_.flatten()}")
print(f"Перетин: {lin_reg.intercept_[0]}")
print("Поліноміальна регресія:")
print(f"Коефіцієнти: {poly_reg.coef_.flatten()}")
print(f"Перетин: {poly_reg.intercept_[0]}")

# Оцінка моделей
mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

print("\nОцінка якості моделей:")
print(f"Лінійна регресія - MSE: {mse_linear:.2f}, R2: {r2_linear:.2f}")
print(f"Поліноміальна регресія - MSE: {mse_poly:.2f}, R2: {r2_poly:.2f}")
