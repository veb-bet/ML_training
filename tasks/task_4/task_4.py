import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
data = pd.read_csv('housing_data.csv')
# Разделение данных на признаки и целевую переменную
X = data[['area', 'bedrooms', 'bathrooms', 'year_built', 'has_garage', 'distance_to_center']]
y = data['price']
# Разделение данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)
# Оценка качества модели на тестовых данных
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Использование обученной модели для предсказания на новых данных
new_data = np.array([[150, 3, 2, 2010, 1, 5]])
predicted_price = model.predict(new_data)
print(f"Предсказанная цена дома: {predicted_price[0]:.2f} долларов")
