import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

# Завантажте зразок набору даних
df = pd.read_csv('data_metrics.csv')
print(df.head())

# Додайте стовпці для прогнозованих міток
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')

# Функції для знаходження TP, FN, FP, TN
def ivanov_find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def ivanov_find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def ivanov_find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def ivanov_find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

# Функція для знаходження всіх значень матриці плутанини
def ivanov_find_conf_matrix_values(y_true, y_pred):
    TP = ivanov_find_TP(y_true, y_pred)
    FN = ivanov_find_FN(y_true, y_pred)
    FP = ivanov_find_FP(y_true, y_pred)
    TN = ivanov_find_TN(y_true, y_pred)
    return TP, FN, FP, TN

# Функція для побудови матриці плутанини
def ivanov_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = ivanov_find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

# Перевірка
print('TP:', ivanov_find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', ivanov_find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', ivanov_find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', ivanov_find_TN(df.actual_label.values, df.predicted_RF.values))

# Перевірка матриці плутанини
print("My Confusion Matrix RF:\n", ivanov_confusion_matrix(df.actual_label.values, df.predicted_RF.values))
print("Sklearn Confusion Matrix RF:\n", confusion_matrix(df.actual_label.values, df.predicted_RF.values))

# Перевірка на відповідність
assert np.array_equal(ivanov_confusion_matrix(df.actual_label.values, df.predicted_RF.values), 
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values)), 'my_confusion_matrix() is not correct for RF'
assert np.array_equal(ivanov_confusion_matrix(df.actual_label.values, df.predicted_LR.values), 
                      confusion_matrix(df.actual_label.values, df.predicted_LR.values)), 'my_confusion_matrix() is not correct for LR'

# Функція для обчислення accuracy_score
def ivanov_my_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = ivanov_find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

# Перевірка на відповідність
assert ivanov_my_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values), 'my_accuracy_score failed on RF'
assert ivanov_my_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values, df.predicted_LR.values), 'my_accuracy_score failed on LR'

# Виведення точності
print('Accuracy RF: %.3f' % ivanov_my_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print('Accuracy LR: %.3f' % ivanov_my_accuracy_score(df.actual_label.values, df.predicted_LR.values))

# Функція для обчислення recall_score
def ivanov_my_recall_score(y_true, y_pred):
    TP, FN, FP, TN = ivanov_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

# Перевірка на відповідність
assert ivanov_my_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values, df.predicted_RF.values), 'my_recall_score failed on RF'
assert ivanov_my_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values, df.predicted_LR.values), 'my_recall_score failed on LR'

# Виведення повноти
print('Recall RF: %.3f' % ivanov_my_recall_score(df.actual_label.values, df.predicted_RF.values))
print('Recall LR: %.3f' % ivanov_my_recall_score(df.actual_label.values, df.predicted_LR.values))

# Функція для обчислення precision_score
def ivanov_my_precision_score(y_true, y_pred):
    TP, FN, FP, TN = ivanov_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

# Перевірка на відповідність
assert ivanov_my_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values, df.predicted_RF.values), 'my_precision_score failed on RF'
assert ivanov_my_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values, df.predicted_LR.values), 'my_precision_score failed on LR'

# Виведення точності
print('Precision RF: %.3f' % ivanov_my_precision_score(df.actual_label.values, df.predicted_RF.values))
print('Precision LR: %.3f' % ivanov_my_precision_score(df.actual_label.values, df.predicted_LR.values))

# Функція для обчислення f1_score
def ivanov_my_f1_score(y_true, y_pred):
    precision = ivanov_my_precision_score(y_true, y_pred)
    recall = ivanov_my_recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

# Виведення F1 оцінки
print('F1 Score RF: %.3f' % ivanov_my_f1_score(df.actual_label.values, df.predicted_RF.values))
print('F1 Score LR: %.3f' % ivanov_my_f1_score(df.actual_label.values, df.predicted_LR.values))

# Підсумкова частина для ROC і ROC-AUC
from sklearn.metrics import roc_curve, roc_auc_score

# Обчислення ROC і AUC
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR)

roc_auc_RF = roc_auc_score(df.actual_label.values, df.predicted_RF.values)
roc_auc_LR = roc_auc_score(df.actual_label.values, df.predicted_LR.values)

print('ROC AUC RF:', roc_auc_RF)
print('ROC AUC LR:', roc_auc_LR)

# Візуалізація ROC-кривої
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr_RF, tpr_RF, color='blue', label='ROC curve RF (area = %0.2f)' % roc_auc_RF)
plt.plot(fpr_LR, tpr_LR, color='red', label='ROC curve LR (area = %0.2f)' % roc_auc_LR)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
