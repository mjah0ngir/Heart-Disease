# 🫀 Heart Disease Prediction — ML Classification Study

> Учебный проект по машинному обучению: предсказание болезни сердца с помощью алгоритмов классификации

---

## 📌 Описание проекта

В данном исследовании решается задача **бинарной классификации**: определить, есть ли у пациента болезнь сердца, на основе клинических показателей.

**Датасет:** UCI Heart Disease Dataset  
**Источник:** [Kaggle / UCI ML Repository](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)  
**Задача:** предсказать `target` (0 = норма, 1 = болезнь сердца)

---

## 📁 Структура репозитория

```
heart-disease-ml/
├── heart_disease_analysis.ipynb   # Jupyter Notebook с полным исследованием
├── heart_disease.csv              # Датасет
├── README.md                      # Описание проекта (этот файл)
└── plots/                         # Графики (генерируются при запуске ноутбука)
    ├── plot_target_distribution.png
    ├── plot_feature_distributions.png
    ├── plot_correlation_matrix.png
    ├── plot_categorical_features.png
    ├── plot_knn_k_selection.png
    ├── plot_model_comparison.png
    ├── plot_confusion_matrices.png
    ├── plot_roc_curves.png
    └── plot_feature_importance.png
```

---

## 📊 Описание данных

| Признак | Тип | Описание |
|---------|-----|----------|
| `age` | int | Возраст пациента (лет) |
| `sex` | binary | Пол: 1 = мужчина, 0 = женщина |
| `cp` | categorical | Тип боли в груди (0–3) |
| `trestbps` | int | Артериальное давление в покое (мм рт. ст.) |
| `chol` | int | Холестерин сыворотки (мг/дл) |
| `fbs` | binary | Сахар натощак > 120 мг/дл (1 = да) |
| `restecg` | categorical | Результаты ЭКГ в покое (0–2) |
| `thalach` | int | Максимальная достигнутая ЧСС |
| `exang` | binary | Стенокардия при нагрузке (1 = да) |
| `oldpeak` | float | Депрессия ST при нагрузке |
| `slope` | categorical | Наклон пика ST нагрузки (0–2) |
| `ca` | int | Число крупных сосудов (0–4) |
| `thal` | categorical | Талассемия: 1=норм, 2=фикс.дефект, 3=обратимый |
| `target` | **binary** | **Целевая переменная** (1 = болезнь, 0 = норма) |

---

## 🤖 Алгоритмы машинного обучения

В проекте применены и сравнены 5 алгоритмов:

| Алгоритм | Подбор параметров | Ключевые гиперпараметры |
|----------|------------------|------------------------|
| **Logistic Regression** | — | `C`, `max_iter` |
| **Decision Tree** | GridSearchCV | `max_depth`, `min_samples_split`, `criterion` |
| **Random Forest** | GridSearchCV | `n_estimators`, `max_depth`, `max_features` |
| **KNN** | Cross-validation по k | `n_neighbors` |
| **SVM** | GridSearchCV | `C`, `kernel`, `gamma` |

---

## 📈 Методология

1. **Загрузка и первичный осмотр** — форма данных, типы, пропуски
2. **EDA (разведочный анализ)**:
   - Распределение классов
   - Гистограммы по признакам (раздельно для классов)
   - Матрица корреляций
   - Анализ категориальных переменных
3. **Очистка данных** — проверка выбросов методом IQR
4. **Подготовка данных**:
   - Train/Test Split (80/20, stratified)
   - StandardScaler для масштабирования
5. **Обучение моделей** с GridSearchCV и кросс-валидацией (5-fold)
6. **Оценка**:
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC-кривые и AUC
   - Важность признаков (Random Forest)

---

## 🚀 Запуск

### Требования

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Запуск ноутбука

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-ml.git
cd heart-disease-ml
jupyter notebook heart_disease_analysis.ipynb
```

---

## 📉 Ключевые выводы

- **Лучший результат** показал **Random Forest** (с подобранными гиперпараметрами)
- **Самые важные признаки** для предсказания: `ca` (число сосудов), `thalach` (макс. ЧСС), `cp` (тип боли), `oldpeak` (депрессия ST)
- Все модели превысили **80% accuracy** на тестовой выборке
- **Logistic Regression** — отличный baseline, результат сравним с более сложными моделями
- **Decision Tree** без ограничений склонен к переобучению, важен подбор `max_depth`

---

## 🧠 Вопросы для защиты

<details>
<summary>Почему нужно масштабировать данные перед KNN и SVM?</summary>

KNN использует евклидово расстояние между точками. Если признаки имеют разный масштаб (например, `age` в диапазоне 30–80 и `chol` в диапазоне 100–600), то `chol` будет доминировать в вычислении расстояния, и модель фактически будет использовать только этот признак. StandardScaler приводит все признаки к среднему 0 и стд. отклонению 1, устраняя этот эффект. SVM по аналогичным причинам также чувствителен к масштабу.
</details>

<details>
<summary>Что такое кросс-валидация и зачем она нужна?</summary>

Кросс-валидация (CV) — метод оценки обобщающей способности модели. При 5-fold CV данные делятся на 5 равных частей; модель обучается 5 раз, каждый раз используя 4 части для обучения и 1 для валидации. Усреднённый результат даёт надёжную оценку качества модели, снижая влияние случайного разбиения на train/test.
</details>

<details>
<summary>Почему Random Forest устойчивее Decision Tree?</summary>

Decision Tree — единственное дерево, которое может переобучиться, запомнив шум в данных. Random Forest строит ансамбль из многих деревьев, каждое обучается на случайной подвыборке данных и признаков (bagging + feature randomness). Итоговое предсказание — голосование всех деревьев, что снижает дисперсию и уменьшает переобучение.
</details>

<details>
<summary>Что такое ROC-кривая и AUC?</summary>

ROC-кривая строится путём варьирования порога классификации от 0 до 1 и отображает TPR (True Positive Rate = Recall) против FPR (False Positive Rate). AUC (Area Under Curve) — площадь под этой кривой: 0.5 = случайный классификатор, 1.0 = идеальный. Чем больше AUC, тем лучше модель различает классы.
</details>

---

## 👤 Автор

Студент, курс: Машинное обучение / Анализ данных  
Выполнено как домашнее задание к семинару
