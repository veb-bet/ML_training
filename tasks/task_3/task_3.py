import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Создаем датасет для сложения однозначных чисел
X = np.random.randint(0, 10, size=(1000, 2))
y = np.sum(X, axis=1)
# Разделяем данные на обучающую и тестовую выборки
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]
# Создаем и обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)
# Оцениваем качество модели на тестовой выборке
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {mse:.2f}")
# Используем обученную модель для предсказания на новых данных
new_data = np.array([[3, 1]])
result = model.predict(new_data)
print(f"Сумма чисел {new_data[0][0]} и {new_data[0][1]} равна {result[0]:.0f}")
