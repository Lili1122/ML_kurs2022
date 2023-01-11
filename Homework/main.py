import pandas as pd


#Загрузим датафрейм и ознакомимся с ним
df = pd.read_csv('train_dataset_train.csv')
df.head()
#Размер датафрейма
print(df.shape)
#список колонок датасета
print(df.columns)

#описательные статистики для всего датасета
print(df.describe())


#После ознакомления с датасетом посмотрим, какие значения принимают переменные с характеристиками.

#Распечатаем в цикле по каждой колонке название колонки, количество уникальных значений, а затем список возможных значений вместе с их количеством появления в датасете.

cols = df.columns
for col in cols:
    print(f"Характеристика: {col}")
    print(f"Количество уникальных значений: {df[col].nunique()}")
    print(f"Список значений: {df[col]}")
    print(df[col])
    print('///////////////////////////////////////////////////')

# Выведем количество полностью заполненных объектов и их процент из всей выборки

values = ((df.isna().sum() / len(df)) * 100).sort_values()
count = 0

for i in values:
    if i == 0:
        count += 1
print(f'Количество полностью заполненных объектов - {count}')
print(f'Их процент из всей выборки - {int(count / len (values) * 100)}%')

#Данные без пропусков!

###ВЫБРОСИМ ЛИШНИЕ СТОЛБЦЫ
columns_to_drop = [
    ############## НЕНУЖНЫЕ
    'id',
    'line_nm',
    'ticket_id',
    'entrance_nm',
    'station_nm',
]

df.drop(columns_to_drop, axis=1, inplace=True)





###РАСШИРИМ КАТЕГОРИАЛЬНЫЕ ПЕРЕМЕННЫЕ С ПОМОЩЬЮ ONEHOTENCODER

from sklearn.preprocessing import OneHotEncoder

columns_to_drop = [
    'ticket_type_nm',
    'entrance_id',
    'station_id',
    'line_id'
]

ohe = OneHotEncoder(sparse=False)
ohe.fit(df[['ticket_type_nm','entrance_id','station_id','line_id']])
ohe_model = ohe.transform(df[['ticket_type_nm','entrance_id','station_id','line_id']])
df[ohe.get_feature_names_out()] = ohe_model
df.drop(columns_to_drop, axis=1, inplace=True)


###ОСТАВИМ В pass_dttm только дни недели и часы

df.pass_dttm = pd.to_datetime(df.pass_dttm)
df['hour'] = df.pass_dttm.apply(lambda x: x.hour)
df['day_of_week'] = df.pass_dttm.dt.weekday
df = df.drop(['pass_dttm'], axis=1)




##########################ФОРМИРОВАНИЕ ML-модели целевые переменные


y_reg = df['time_to_under']
y_class = df['label']
X = df.drop(['time_to_under','label'], axis=1)
from sklearn.model_selection import train_test_split

###ЛИНЕЙНАЯ РЕГРЕССИЯ ДЛЯ ПРЕДСКАЗАНИЯ time_to_under
from sklearn.linear_model import LinearRegression
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg,
                                                    test_size=0.33,
                                                    random_state=True)


reg = LinearRegression().fit(X_train, y_reg_train)
y_reg_pred = reg.predict(X_test)

###КЛАССИФИКАЦИЯ С ПОМОЩЬЮ РАНДОМ ФОРЕСТ
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class,
                                                    test_size=0.33,
                                                    random_state=True)

# Создаём модель леса из 10 деревьев
mrandfor = RandomForestClassifier(n_estimators=10)

# Обучаем на тренировочных данных
mrandfor.fit(X_train, y_class_train)
# Предсказания
y_class_pred = mrandfor.predict(X_test)


########ИТОГОВАЯ МЕТРИКА:
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score

print(0.5 * (r2_score(y_reg_test, y_reg_pred) + recall_score(y_class_test, y_class_pred, average='weighted')))





