import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. Генерація випадкових даних ---
np.random.seed(42)
n = 1000

# Незалежна змінна — кількість відвідувань
visits = np.random.randint(1, 100, n)

# Створимо нелінійну залежність:
# Користувачі, які частіше відвідують бібліотеку, мають тенденцію читати більше сторінок,
# але з певним насиченням (логістичний/експоненційний ефект)
def nonlinear_func(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c  # типова форма насичення

# Створення даних із шумом
pages = nonlinear_func(visits, a=200, b=0.05, c=10) + np.random.normal(0, 10, n)

# --- 2. Побудова DataFrame ---
data = pd.DataFrame({
    'Visits': visits,
    'Pages_Per_Session': pages
})

# --- 3. Оцінка параметрів нелінійної моделі ---
# Початкові оцінки параметрів
initial_guess = [150, 0.03, 5]
params, covariance = curve_fit(nonlinear_func, data['Visits'], data['Pages_Per_Session'], p0=initial_guess)

a_est, b_est, c_est = params
print(f"Оцінені параметри моделі: a = {a_est:.3f}, b = {b_est:.4f}, c = {c_est:.3f}")

# --- 4. Побудова прогнозу ---
predicted_pages = nonlinear_func(data['Visits'], *params)

# --- 5. Оцінка якості моделі ---
r2 = r2_score(data['Pages_Per_Session'], predicted_pages)
rmse = np.sqrt(mean_squared_error(data['Pages_Per_Session'], predicted_pages))

print(f"R² = {r2:.4f}")
print(f"RMSE = {rmse:.3f}")

# --- 6. Візуалізація ---
plt.figure(figsize=(8, 6))
plt.scatter(data['Visits'], data['Pages_Per_Session'], color='blue', alpha=0.4, label='Фактичні дані')
plt.plot(np.sort(data['Visits']),
         nonlinear_func(np.sort(data['Visits']), *params),
         color='red', linewidth=2, label='Нелінійна модель')

plt.title('Нелінійна залежність: Кількість відвідувань → Кількість прочитаних сторінок')
plt.xlabel('Кількість відвідувань користувача')
plt.ylabel('Середня кількість прочитаних сторінок за сеанс')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 7. Збереження ---
data.to_csv("nonlinear_library_model.csv", index=False)
print("Дані збережено у файл 'nonlinear_library_model.csv'")
