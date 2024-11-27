import numpy as np  
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Завантажуємо дані з файлу
data = np.loadtxt('data_multivar_regr.txt', delimiter=',')

# Розділяємо дані на ознаки (X) та цільову змінну (y)
X = data[:, :-1]
y = data[:, -1]

# Розділяємо дані на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Лінійна регресія
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

# Прогноз для тестового набору
y_pred_linear = linear_regressor.predict(X_test)

# Метрики якості лінійної регресії
print("Linear Regressor performance:")
print(f"Mean absolute error: {sm.mean_absolute_error(y_test, y_pred_linear)}")
print(f"Mean squared error: {sm.mean_squared_error(y_test, y_pred_linear)}")
print(f"Median absolute error: {sm.median_absolute_error(y_test, y_pred_linear)}")
print(f"Explained variance score: {sm.explained_variance_score(y_test, y_pred_linear)}")
print(f"R2 score: {sm.r2_score(y_test, y_pred_linear)}")

# Поліноміальна регресія (степінь 10)
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

# Створюємо та навчаємо лінійну модель для поліноміальних даних
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

# Прогноз для тестового набору на основі поліноміальної регресії
X_test_transformed = polynomial.transform(X_test)
y_pred_poly = poly_linear_model.predict(X_test_transformed)

# Метрики якості поліноміальної регресії
print("\nPolynomial Regressor performance:")
print(f"Mean absolute error: {sm.mean_absolute_error(y_test, y_pred_poly)}")
print(f"Mean squared error: {sm.mean_squared_error(y_test, y_pred_poly)}")
print(f"Median absolute error: {sm.median_absolute_error(y_test, y_pred_poly)}")
print(f"Explained variance score: {sm.explained_variance_score(y_test, y_pred_poly)}")
print(f"R2 score: {sm.r2_score(y_test, y_pred_poly)}")

# Прогноз для окремої точки даних
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)
print("\nLinear regression for datapoint:", linear_regressor.predict(datapoint))
print("Polynomial regression for datapoint:", poly_linear_model.predict(poly_datapoint))
