import numpy as np

# Змінна input_data відповідно до вашого варіанту
input_data = np.array([
    4.1, -5.9, -3.5, -1.9, 4.6, 3.9, -4.2, 6.8,
    6.3, 3.9, 3.4, 1.2, 3.2
])  # Приклад для варіанту 2

# Параметри
threshold = 0  # Поріг бінаризації
mean_exclude = np.mean(input_data)

# Бінаризація
binary_data = (input_data > threshold).astype(int)

# Виключення середнього
mean_excluded_data = input_data - mean_exclude

# Масштабування
scaled_data = (mean_excluded_data - np.min(mean_excluded_data)) / (np.max(mean_excluded_data) - np.min(mean_excluded_data))

# Нормалізація
normalized_data = (scaled_data - np.mean(scaled_data)) / np.std(scaled_data)

# Виведення результатів
print("Вхідні дані:", input_data)
print("Бінаризовані дані:", binary_data)
print("Дані без середнього:", mean_excluded_data)
print("Масштабовані дані:", scaled_data)
print("Нормалізовані дані:", normalized_data)
