import numpy as np
from sklearn.linear_model import LinearRegression

# Создание примерных данных о погоде
X_train = np.array([[10, 20, 5], [15, 25, 3], [12, 18, 7], [18, 22, 4], [14, 23, 6]])
y_train = np.array([18, 22, 20, 24, 21])

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание температуры на новый день
new_data = np.array([[16, 24, 5]])
predicted_temp = model.predict(new_data)
print(f"Предсказанная температура: {predicted_temp[0]:.2f} градусов")
