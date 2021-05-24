import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
from matplotlib import pyplot as plt
file = 'data_dev.csv'
data = pd.read_csv(file,sep = ';')
a = data.isna().sum()
b = data.info()
c = data.head(10)
d = data.describe()
data['first_loan'].fillna(data['first_loan'].mean(),inplace = True)#заполняем NaN значения средними значениями
clear_data = data.drop(['order_date','order_id','client_id'],axis = 1)#отбрасываем колонки, не имеющие значения для классификации
#Разделение на тестовую и обучающие выборки
y , X = clear_data['expert'], clear_data.drop('expert',axis = 1, inplace=False)
X_train , X_test , y_train , y_test = train_test_split(X , y,
                                                       train_size=0.75,
                                                       random_state=101,
                                                       stratify=y)
#Стандартизируем значения выборки
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
y_train_scaled = np.array(y_train)
#Используем метод главных компонент для уменьшения размера данных
pca_test = PCA(n_components=14)
pca_test.fit(X_train_scaled)
sns.set(style='whitegrid')
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.axvline(linewidth=4, color='r', linestyle = '--', x=9, ymin=0, ymax=1)
plt.show()
evr = pca_test.explained_variance_ratio_
cvr = np.cumsum(pca_test.explained_variance_ratio_)
pca_df = pd.DataFrame()
pca_df['Cumulative Variance Ratio'] = cvr
pca_df['Explained Variance Ratio'] = evr
#Исходя из построенного графика 9 компонент объясняют 90% дисперсии
pca = PCA(n_components=9)
pca.fit(X_train_scaled)
X_train_scaled_pca = pca.transform(X_train_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)
#Для классификации используем Random Forest Classifier 
rfc = RandomForestClassifier()
rfc.fit(X_train_scaled_pca, y_train)
print(rfc.score(X_train_scaled_pca, y_train))
y_pred = rfc.predict(X_test_scaled_pca)
y_probs = rfc.predict_proba(X_test_scaled_pca)[:, 1]
#Определения эффективности модели
conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred),
                                    index = ['actual 0', 'actual 1'],
                                    columns = ['predicted 0', 'predicted 1'])
print('Confusion matrix ',conf_matrix)#Оцениваем матрицу ошибок
print('Baseline Random Forest recall score', recall_score(y_test, y_pred))#Оцениваем полноту
print('Baseline Random Forest precision score', precision_score(y_test, y_pred))#Оцениваем точность
fpr , tpr , t = roc_curve(y_test,y_probs)
print('Area Under the ROC AUC ',roc_auc_score(y_test, y_probs))#Оцениваем площадь под ROC кривой
print('GINI coefficient ',2 * roc_auc_score(y_test, y_probs) - 1)#Расчитываем коэффициент Джинни
sns.lineplot(fpr,tpr)#Строим ROC кривую для наглядности
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.fill_between([0]+fpr.tolist(), [0]+tpr.tolist(),alpha=0.3)
#     В результате:
#  Была построена модель классификации на основе классификатора Random Forest:
#1) Произведена первичная оценка датасета. Выявлены отсутствующие значения в столбце 'first_loan'
#2) Отсутствующие значения столбца 'first_loan' заполнены средними значениями этого столбца.
#    Удалены столбцы 'order_date','order_id','client_id', т.к. они не несут в себе информации необходимой для классификации
#3) Датасет разделен на тестовую и обучающие выборки в соотношении 1 к 3
#4) Произведена стандартизация данных. !NB Деревья принятия решений не требуют стандартизации данных,
#    однако я сталкивался и с обратным мнением. Поэтом я принял решение использовать стандартизированыые данные,
#    т.к. данный подход не должен отрицательно повлиять на качество модели
#5) Использован метод главных компонент для уменьшения размеров данных.
#    Исходя из построенного графика 9 компонент объясняют 90% дисперсии.
#6) Использован алгоритм Random Forest Classificator. Для обучения и теста использовались 9 компонент выборки
#   Произведена оценка модели:
# 1) Оценена матрица ошибок. Результат:
#                                       3426     711
#                                       460      11904
# 2) Произведна оценка точности и полноты
#     Точность 0.9427946233197875
#     Полнота 0.9659442724458205
# 3) Произведена оценка площади ROC кривой.
#     Площадь под ROC кривой 0.9599269700572056
# 4) Расчитан коэффициент Джини:
#     Статистический коэффициент GINI 0.9198539401144112
#   Результат оценки модели. Выводы.
# Результаты вычисления полноты показали, что модель идентифицирует 96% отказов заявок (data['expert'] == 1)
# Расчёт площади под ROC кривой(0.9) показывает, отличное качество модели. (Лекция "Логистическая регрессия". Конспект. Университет ИТМО. СПБ.2019г. стр.25)
# Расчёт коэфициента Джини (0.91) показывает, что результат классификации тяготеет к паре True Positive - False Positive (Симинары по решающим деревьям. Евгений Соколов. 2013г.стр3. http://www.machinelearning.ru/wiki/images/8/89/Sem3_trees.pdf)

#     Примечание:
# В ходе изучения теоретических материалов, я столкнулся со статьёй на habr Дмитрия Горелова (datasanta) сотрудника Devim https://habr.com/ru/post/474062/,
# а также Системы кредитного скоринга. Матричный подход. Артём ТКАЧЁВ, Алексей ШИПУНОВ https://www.nbrb.by/bv/articles/10671.pdf
# в которых говориться о необходимости сегментации датасета на выборки с заявками на различные суммы кредитов.
# Однако в результате вычислений модель показала хороший результат и без разделения датасета на сегменты.