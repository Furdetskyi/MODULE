import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- 1. Генерація випадкових даних ---
np.random.seed(42)
n = 1000

# Незалежні змінні
visits = np.random.randint(1, 50, n)                # X1 — кількість відвідувань
session_time = np.random.uniform(5, 60, n)          # X2 — середня тривалість сеансу
searches = np.random.randint(0, 20, n)              # X3 — кількість пошуків

# --- 2. Формування залежної змінної (із шумом) ---
# Припустимо, що:
# - збільшення кількості відвідувань і часу в системі підвищує кількість сторінок;
# - кількість пошуків має позитивний, але слабший ефект.
pages = 5 + 2.5 * visits + 1.8 * session_time + 0.9 * searches + np.random.normal(0, 25, n)

# --- 3. Формування DataFrame ---
data = pd.DataFrame({
    'Visits': visits,
    'Session_Time': session_time,
    'Searches': searches,
    'Pages_Read': pages
})

# --- 4. Побудова лінійної регресійної моделі ---
X = data[['Visits', 'Session_Time', 'Searches']]
X = sm.add_constant(X)  # додаємо константу
y = data['Pages_Read']

model = sm.OLS(y, X).fit()

# --- 5. Вивід результатів ---
print(model.summary())

# --- 6. Побудова рівняння регресії ---
coeff = model.params
equation = f"Y = {coeff[0]:.2f} + {coeff[1]:.2f}*X1 + {coeff[2]:.2f}*X2 + {coeff[3]:.2f}*X3"
print("\nМатематичне рівняння регресії:")
print(equation)

# --- 7. Візуалізація впливу головної змінної (Visits) ---
plt.scatter(data['Visits'], data['Pages_Read'], alpha=0.4, color='blue', label='Фактичні дані')
predicted = model.predict(X)
plt.scatter(data['Visits'], predicted, color='red', alpha=0.4, label='Прогнозовані значення')
plt.xlabel('Кількість відвідувань користувача')
plt.ylabel('Середня кількість прочитаних сторінок')
plt.title('Регресійна модель: вплив відвідувань на активність користувача')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
