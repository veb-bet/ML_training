import pandas as pd
from sklearn.linear_model import LinearRegression
# Загрузка данных из файла CSV
data = pd.read_csv('auto.csv', encoding='utf-8')
# Разделение данных на признаки и целевую переменную
X = data[['Year', 'Mileage', 'Engine']]
y = data['Price']
# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X, y)
# Вывод информации о модели
print("Коэффициенты линейной регрессии:")
print(model.coef_)
print("Свободный член линейной регрессии:")
print(model.intercept_)
print("Коэффициент детерминации (R^2):")
print(model.score(X, y))
# Пример предсказания цены автомобиля
new_car = [[2018, 50000, 2.0]]
predicted_price = model.predict(new_car)[0]
print(f"Предсказанная цена автомобиля: {predicted_price:.2f} рублей")

