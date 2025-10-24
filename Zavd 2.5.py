import random
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# 1. Генерація випадкових "запитів користувачів"
# -----------------------------
keywords = ['книга', 'читати', 'завантажити', 'користувач', 'електронна', 'бібліотека', 
            'новинка', 'жанр', 'автор', 'рейтинг']

# Створюємо 100 випадкових запитів (по 3-5 слів кожен)
documents = []
for _ in range(100):
    doc = " ".join(random.choices(keywords, k=random.randint(3,5)))
    documents.append(doc)

print("Приклад документів (запитів користувачів):")
print(documents[:5])

# -----------------------------
# 2. Обчислення TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

# Обчислимо середню TF-IDF для кожного слова по всіх документах
avg_tfidf = tfidf_matrix.mean(axis=0).A1
tfidf_scores = dict(zip(feature_names, avg_tfidf))

# Виведемо топ-5 найважливіших слів
top_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nТоп-5 слів за TF-IDF:")
for word, score in top_words:
    print(f"{word}: {score:.4f}")

# -----------------------------
# 3. Закон Ципфа: частота слів
# -----------------------------
# Рахуємо частоту слів у всіх документах
all_words = " ".join(documents).split()
word_counts = Counter(all_words)

# Ранжуємо слова за частотою
sorted_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
ranks = range(1, len(sorted_counts)+1)
frequencies = [freq for word, freq in sorted_counts]

# -----------------------------
# 4. Візуалізація закону Ципфа
# -----------------------------
plt.figure(figsize=(8,5))
plt.loglog(ranks, frequencies, marker="o")
plt.title("Закон Ципфа для слів запитів користувачів")
plt.xlabel("Ранг слова")
plt.ylabel("Частота слова")
plt.grid(True, which="both", ls="--")
plt.show()
