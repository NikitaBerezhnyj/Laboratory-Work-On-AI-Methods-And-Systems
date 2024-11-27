import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантаження даних
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розподіл на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Створення моделі лінійної регресії
regr = linear_model.LinearRegression()

# Навчання моделі
regr.fit(X_train, y_train)

# Прогноз для тестової вибірки
y_pred = regr.predict(X_test)

# Розрахунок коефіцієнтів регресії та показників
print("Коефіцієнти регресії:", regr.coef_)
print("Перетин (intercept):", regr.intercept_)
print("R2 (коефіцієнт кореляції):", r2_score(y_test, y_pred))
print("Середня абсолютна помилка (MAE):", mean_absolute_error(y_test, y_pred))
print("Середньоквадратична помилка (MSE):", mean_squared_error(y_test, y_pred))

# Побудова графіка
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.title('Лінійна регресія: Виміряно vs Передбачено')
plt.show()
