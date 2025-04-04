import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np


data = pd.read_csv('mxmh_survey_results.csv')

data = pd.read_csv('mxmh_survey_results.csv')

# Преобразование "Hours per day" в "Hours per week" (предполагая 7 дней)
data['Hours per week'] = data['Hours per day'] * 7

leq4 = data[data['Hours per day'] <= 4]['Anxiety'].dropna()  # ≤5 часов
geq4 = data[data['Hours per day'] > 4]['Anxiety'].dropna()     # >5 часов
'''
plt.figure(figsize=(8, 4))
sns.histplot(data['Hours per day'], bins=20, kde=True)
plt.title('Распределение часов музыки в день')
plt.show()
'''
def describe_stats(data, name):
    return pd.Series({
        'Минимум': round(np.min(data), 3),
        'Максимум': round(np.max(data), 3),
        'Размах': round(np.ptp(data), 3),
        'Среднее': round(np.mean(data), 3),
        'Дисперсия': round(np.var(data, ddof=1), 3),
        'Ст. отклонение': round(np.std(data, ddof=1), 3),
        'Асимметрия': round(stats.skew(data), 3),
        'Медиана': round(np.median(data), 3),
        'Q1': round(np.percentile(data, 25), 3),
        'Q3': round(np.percentile(data, 75), 3),
        'IQR': round(np.percentile(data, 75) - np.percentile(data, 25), 3)
    }, name=name)


stats_df = pd.concat([
    describe_stats(leq4, '≤4 часов/день'),
    describe_stats(geq4, '>4 часов/день')
], axis=1)
print(stats_df)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(leq4, bins=10, kde=True, color='blue')
plt.title('Тревожность (≤4 часов/день)')

plt.subplot(1, 2, 2)
sns.histplot(geq4, bins=10, kde=True, color='red')
plt.title('Тревожность (>4 часов/день)')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=['≤4 часов']*len(leq4) + ['>4 часов']*len(geq4),
            y=pd.concat([leq4, geq4]))
plt.title('Распределение тревожности')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
stats.probplot(leq4, dist='norm', plot=plt)
plt.title('QQ-plot (≤4 часов)')

plt.subplot(1, 2, 2)
stats.probplot(geq4, dist='norm', plot=plt)
plt.title('QQ-plot (>4 часов)')
plt.show()

plt.figure(figsize=(10, 6))
sns.ecdfplot(data=leq4, label='≤5 часов/день', linewidth=2)
sns.ecdfplot(data=geq4, label='>5 часов/день', linewidth=2)

plt.title('Эмпирическая функция распределения (ECDF) тревожности', fontsize=14)
plt.xlabel('Уровень тревожности', fontsize=12)
plt.ylabel('Вероятность', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(np.arange(0, 11, 1))

plt.tight_layout()
plt.show()
