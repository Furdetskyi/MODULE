import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- 1. Генерація випадкових даних ---
np.random.seed(42)  # для відтворюваності результатів
n = 1000  # кількість користувачів

# Кількість відвідувань користувача (1–100)
visits = np.random.randint(1, 101, n)

# Середня тривалість сеансу (хвилини) з кореляцією до visits
# Наприклад, користувачі, які частіше відвідують бібліотеку, читають довше
session_time = 2.5 * visits + np.random.normal(0, 20, n) + 30

# --- 2. Формування датафрейму ---
data = pd.DataFrame({
    'User_ID': range(1, n + 1),
    'Visits': visits,
    'Avg_Session_Time': session_time
})

# --- 3. Обчислення коефіцієнта кореляції ---
corr_value, p_value = pearsonr(data['Visits'], data['Avg_Session_Time'])

# --- 4. Вивід результатів ---
print(f"Коефіцієнт кореляції між кількістю відвідувань і середньою тривалістю сеансу: {corr_value:.3f}")
print(f"P-значення: {p_value:.5f}")

# --- 5. Візуалізація ---
plt.figure(figsize=(8, 6))
plt.scatter(data['Visits'], data['Avg_Session_Time'], alpha=0.5, color='royalblue')
plt.title('Залежність активності користувача від кількості відвідувань')
plt.xlabel('Кількість відвідувань користувача')
plt.ylabel('Середня тривалість сеансу (хв)')
plt.grid(True, linestyle='--', alpha=0.6)

# Додатково — лінія тренду
m, b = np.polyfit(data['Visits'], data['Avg_Session_Time'], 1)
plt.plot(data['Visits'], m * data['Visits'] + b, color='red', linewidth=2, label='Тренд')
plt.legend()
plt.show()

# --- 6. Збереження у файл ---
data.to_csv("library_user_activity.csv", index=False)
print(" Дані збережено у файл 'library_user_activity.csv'")
