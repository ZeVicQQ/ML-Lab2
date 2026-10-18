import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

# 1. Генерация случайной задачи регрессии
X, y, true_coef = make_regression(
    n_samples=500, n_features=100, n_informative=10, 
    noise=15.0, coef=True, random_state=42
)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Модель линейной регрессии (OLS)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"Linear Regression R^2: {r2_score(y_test, y_pred_lr):.4f}")

# 3. Модель гребневой регрессии (Ridge)
alphas_ridge = [0.1, 1, 10, 100]
ridge_models = {}
print("\nRidge Regression:")
for alpha in alphas_ridge:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_models[alpha] = ridge
    print(f"Alpha={alpha}: R^2 = {r2_score(y_test, ridge.predict(X_test)):.4f}")

# Выбор оптимального alpha для Ridge (например, по R^2 на тесте)
best_alpha_ridge = max(alphas_ridge, key=lambda a: r2_score(y_test, ridge_models[a].predict(X_test)))
print(f"Оптимальный alpha для Ridge: {best_alpha_ridge}")

# 4. Модель лассо-регрессии (Lasso)
alphas_lasso = [0.0001, 0.01, 1, 10]
lasso_models = {}
print("\nLasso Regression:")
for alpha in alphas_lasso:
    # Увеличен max_iter для сходимости
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_models[alpha] = lasso
    print(f"Alpha={alpha}: R^2 = {r2_score(y_test, lasso.predict(X_test)):.4f}, Нулевых коэффициентов: {np.sum(lasso.coef_ == 0)}")

best_alpha_lasso = max(alphas_lasso, key=lambda a: r2_score(y_test, lasso_models[a].predict(X_test)))
print(f"Оптимальный alpha для Lasso: {best_alpha_lasso}")

# 5. Построение графиков
plt.figure(figsize=(18, 5))

# График 1: Сравнение коэффициентов (Линейная vs Ridge)
plt.subplot(1, 3, 1)
plt.plot(lr.coef_, 'o', label='Линейная регрессия', alpha=0.6)
plt.plot(ridge_models[10].coef_, '^', label='Ridge alpha=10')
plt.plot(ridge_models[1].coef_, 'v', label='Ridge alpha=1')
plt.xlabel('Индекс коэффициента')
plt.ylabel('Оценка коэффициента')
plt.legend()
plt.title('Сравнение коэффициентов: OLS vs Ridge')

# График 2: Сравнение коэффициентов (Ridge vs Lasso)
plt.subplot(1, 3, 2)
plt.plot(ridge_models[10].coef_, '^', label='Ridge alpha=10', alpha=0.6)
plt.plot(lasso_models[1].coef_, 'v', label='Lasso alpha=1')
plt.plot(lasso_models[0.01].coef_, 'o', label='Lasso alpha=0.01')
plt.xlabel('Индекс коэффициента')
plt.ylabel('Оценка коэффициента')
plt.legend()
plt.title('Сравнение коэффициентов: Ridge vs Lasso')

# График 3: Кривая обучения (Learning Curve)
plt.subplot(1, 3, 3)
train_sizes, train_scores, test_scores = learning_curve(
    Ridge(alpha=best_alpha_ridge), X, y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Обучение (Ridge)")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Тест (Ridge)")
plt.xlabel('Размер обучающей выборки')
plt.ylabel('R^2 Score')
plt.legend()
plt.title(f'Кривая обучения (Ridge, alpha={best_alpha_ridge})')

plt.tight_layout()
plt.show()