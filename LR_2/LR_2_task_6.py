import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Варіант 1 (як у завданні)
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Функція для побудови кривих навчання
def plot_learning_curves(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y.ravel(), 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='neg_mean_squared_error'
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Mean Squared Error')
    plt.grid()
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    
    plt.legend(loc='best')
    plt.show()

# Моделі для аналізу
linear_model = make_pipeline(LinearRegression())
quadratic_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
high_degree_model = make_pipeline(PolynomialFeatures(degree=10), LinearRegression())

# Побудова кривих навчання для різних моделей
print("Криві навчання для лінійної моделі:")
plot_learning_curves(linear_model, X, y, 'Learning Curves (Linear Regression)')

print("\nКриві навчання для квадратичної моделі:")
plot_learning_curves(quadratic_model, X, y, 'Learning Curves (Quadratic Regression)')

print("\nКриві навчання для поліноміальної моделі 10-го ступеня:")
plot_learning_curves(high_degree_model, X, y, 'Learning Curves (Polynomial Regression, degree=10)')