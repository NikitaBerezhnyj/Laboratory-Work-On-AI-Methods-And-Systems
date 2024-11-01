import numpy as np
import argparse

class Perceptron:
    def __init__(self, input_size, learning_rate=0.2, epochs=10):
        # Ініціалізація ваг та параметрів навчання
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    # Функція активації
    def step_function(self, value):
        return 1 if value >= 0 else 0
    
    def predict(self, x):
        # Прогнозування виходу на основі вхідних даних
        weighted_sum = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.step_function(weighted_sum), weighted_sum
    
    def fit(self, X, y, show_table):
        if show_table:
            # Виведення заголовка таблиці навчання
            print(' ' + '-' * 162)
            print(f"|  {'w1':^6}  |  {'w2':^6}  |  {'w3':^6}  |  {'O':^6}  |  {'x1':^5}  |  {'x2':^5}  |  {'x3':^5}  |  {'a':^6}  |  {'Y':^5}  |  {'T':^5}  |  {'η(T - Y)':^6}   |  {'δw1':^6}  |  {'δw2':^6}  |  {'δw3':^6}  |  {'δO':^6}  |")
            print(' ' + '-' * 162)
        
        for _ in range(self.epochs):
            for i in range(len(X)):
                prediction, weighted_sum = self.predict(X[i])
                error = y[i] - prediction  # Обчислення помилки
                
                # Оновлення ваг на основі помилки
                delta_w = self.learning_rate * error * X[i]
                delta_o = self.learning_rate * error
                
                w1_old, w2_old, w3_old = self.weights[1:]
                o_old = self.weights[0]
                
                self.weights[1:] += delta_w
                self.weights[0] += delta_o

                # Виведення даних навчання в таблиці
                if show_table:
                    print(f"|  {w1_old:^6.2f}  |  {w2_old:^6.2f}  |  {w3_old:^6.2f}  |  {o_old:^6.2f}  |  {X[i][0]:^5}  |  {X[i][1]:^5}  |  {X[i][2]:^5}  |  {weighted_sum:^6.2f}  |  {prediction:^5}  |  {y[i]:^5}  |  {self.learning_rate * error:^9}  |  {delta_w[0]:^6.2f}  |  {delta_w[1]:^6.2f}  |  {delta_w[2]:^6.2f}  |  {delta_o:^6.2f}  |")
            if show_table:
                print(' ' + '-' * 162)

if __name__ == "__main__":
    # Налаштування аргументів командного рядка
    parser = argparse.ArgumentParser(description="Training a perceptron for a logic function (x1 ∧ x2) ∨ x3 with additional table output.")
    parser.add_argument('-t', '--table', action='store_true', help="Show learning table")
    args = parser.parse_args()

    # Визначення вхідних даних та очікуваних виходів
    X = np.array([
        [0, 0, 0], 
        [0, 0, 1], 
        [0, 1, 0], 
        [0, 1, 1], 
        [1, 0, 0], 
        [1, 0, 1], 
        [1, 1, 0], 
        [1, 1, 1]])

    y = np.array([0, 1, 0, 1, 0, 1, 1, 1])

    perceptron = Perceptron(input_size=3)
    perceptron.fit(X, y, args.table)

    if not args.table:
        # Виведення результатів прогнозування без таблиці
        for i in range(len(X)):
            print(f"Input: {X[i]}, Predicted: {perceptron.predict(X[i])[0]}, Actual: {y[i]}")
