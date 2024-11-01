import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Завантаження даних
data = pd.read_csv('data_multivar_nb.txt', header=None)  # Налаштуйте заголовок, якщо потрібно
X = data.iloc[:, :-1]  # Ознаки (всі колонки, окрім останньої)
y = data.iloc[:, -1]   # Мітки (остання колонка)

# Розбивка даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Тренування та оцінка класифікатора SVM
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

# Оцінка SVM
print("Звіт класифікації SVM:")
print(classification_report(y_test, y_pred_svm))
print("Точність SVM:", accuracy_score(y_test, y_pred_svm))

# Тренування та оцінка наївного байєсовського класифікатора
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

# Оцінка наївного байєсовського класифікатора
print("Звіт класифікації наївного байєсовського класифікатора:")
print(classification_report(y_test, y_pred_nb))
print("Точність наївного байєсовського класифікатора:", accuracy_score(y_test, y_pred_nb))