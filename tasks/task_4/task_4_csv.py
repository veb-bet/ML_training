import numpy as np
import pandas as pd

# Генерация случайных данных
num_samples = 1000
area = np.random.uniform(50, 300, size=num_samples)
bedrooms = np.random.randint(1, 7, size=num_samples)
bathrooms = np.random.randint(1, 5, size=num_samples)
year_built = np.random.randint(1950, 2023, size=num_samples)
has_garage = np.random.randint(0, 2, size=num_samples)
distance_to_center = np.random.uniform(1, 20, size=num_samples)
price = 50000 * area + 20000 * bedrooms + 30000 * bathrooms + 500 * (2022 - year_built) + 50000 * has_garage - 5000 * distance_to_center + np.random.normal(0, 20000, size=num_samples)
# Создание DataFrame
data = pd.DataFrame({
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'year_built': year_built,
    'has_garage': has_garage,
    'distance_to_center': distance_to_center,
    'price': price
})
# Сохранение данных в CSV-файл
data.to_csv('housing_data.csv', index=False)
