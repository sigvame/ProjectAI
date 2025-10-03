import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


#завдання 1

# num_days = 30
# start_date = pd.to_datetime('2025-01-01')
# dates = pd.date_range(start=start_date, periods=num_days, freq='D')
# np.random.seed(42)
#
# b_users = np.linspace(100, 150, num_days)
# users = (b_users + np.random.normal(0, 10, num_days)).astype(int)
# sessions = (users * 1.5 + np.random.normal(0, 15, num_days)).astype(int)
# r = (sessions * 0.8 + np.random.normal(0, 10, num_days))
#
# df = pd.DataFrame({
#     'date': dates,
#     'users': users,
#     'sessions': sessions,
#     'revenue': r
# })
#
# print("DataFrame:")
# print(df.head())
# print("\n" + "=" * 50 + "\n")
#
# matrix = df[['users', 'sessions', 'revenue']].corr()
# print("Кореляційна матриця:")
# print(matrix)
# print("\n" + "=" * 50 + "\n")
#
# fig, l = plt.subplots(1, 3, figsize=(18, 5))
# fig.suptitle('Діаграми розсіювання', fontsize=16)
#
# # users-sessions
# l[0].scatter(df['users'], df['sessions'], color='blue', alpha=0.6)
# l[0].set_title('Користувачі - Сесії')
# l[0].set_xlabel('Користувачі')
# l[0].set_ylabel('Сесії')
# l[0].grid(True, linestyle=':', alpha=0.5)
#
# # users-revenue
# l[1].scatter(df['users'], df['revenue'], color='red', alpha=0.6)
# l[1].set_title('Користувачі - Виручка')
# l[1].set_xlabel('Користувачі')
# l[1].set_ylabel('Виручка')
# l[1].grid(True, linestyle=':', alpha=0.5)
#
# #sessions-revenue
# l[2].scatter(df['sessions'], df['revenue'], color='green', alpha=0.6)
# l[2].set_title('Сесії - Виручка')
# l[2].set_xlabel('Сесії')
# l[2].set_ylabel('Виручка')
# l[2].grid(True, linestyle=':', alpha=0.5)
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=4.0)
# plt.show()
#
#
# plt.figure(figsize=(12, 6))
# plt.plot(df['date'], df['revenue'], marker='o', linestyle='-', color='purple', label='Виручка')
#
# plt.title('Лінійний графік за датами')
# plt.xlabel('Дата')
# plt.ylabel('Виручка')
# plt.xticks(rotation=45)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()




#завдання 2

# N = 200
#
# data = {
#     'Group': ['A'] * N + ['B'] * N,
#     'Converted': np.concatenate([
#         np.random.choice([0, 1], size=N, p=[0.90, 0.10]), # 10% для A
#         np.random.choice([0, 1], size=N, p=[0.88, 0.12])  # 12% для B
#     ])
# }
#
# df = pd.DataFrame(data)
#
# summary = df.groupby('Group')['Converted'].agg(['count', 'sum']).reset_index()
# summary.columns = ['Group', 'Total', 'Conversions']
# summary['CR'] = summary['Conversions'] / summary['Total']
#
# cr_A = summary[summary['Group'] == 'A']['CR'].values[0]
# cr_B = summary[summary['Group'] == 'B']['CR'].values[0]
# difference = cr_B - cr_A
# change = difference / cr_A
#
# z = stats.norm.ppf(0.975)
#
# def calc_ci(row):
#     p = row['CR']
#     n = row['Total']
#     s = np.sqrt((p * (1 - p)) / n)
#     m = z * s
#     return pd.Series([p - m, p + m])
#
# summary[['CI_L', 'CI_U']] = summary.apply(calc_ci, axis=1)
#
# print(summary[['Group', 'Total', 'Conversions', 'CR', 'CI_L', 'CI_U']].to_string(index=False, float_format="%.4f"))
# print("\n" + "=" * 50)
# print(f"Абсолютна різниця: {difference:.4f}")
# print(f"Відносна зміна: {change:.2%}")
# print("=" * 50 + "\n")
#
#
# plt.figure(figsize=(8, 6))
# bars = plt.bar(summary['Group'], summary['CR'], color=['#4C72B0', '#DD8452'])
#
# plt.ylabel("Конверсія")
# plt.title("Конверсія груп A та B")
# plt.ylim(bottom=0, top=summary['CR'].max() * 1.2) # Збільшуємо верхню межу для міток
# plt.grid(axis='y', linestyle='--', alpha=0.6)
#
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
#              f'{height:.2%}',
#              ha='center', va='bottom', fontsize=12, fontweight='bold')
#
# plt.show()


#завдання 3

#завдання 4

NUM_DAYS = 90
win = 7

start_date = pd.to_datetime('2025-10-01')
dates = pd.date_range(start=start_date, periods=NUM_DAYS, freq='D')
np.random.seed(42)

trend = np.linspace(100, 150, NUM_DAYS)
noise = np.random.normal(0, 10, NUM_DAYS)

sales = trend + noise
df = pd.DataFrame({
    'date': dates,
    'sales': sales
})
df.set_index('date', inplace=True)
df['Rolling_Mean'] = df['sales'].rolling(window=win).mean()
df['Rolling_Std'] = df['sales'].rolling(window=win).std()

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['sales'], label='Вихідні продажі', color='lightgrey', alpha=0.8)

plt.plot(df.index, df['Rolling_Mean'], label=f'Ковзне середнє (Вікно {win})', color='blue', linewidth=2)

plt.title(f'Продажі та ковзне середнє (Вікно {win} днів)')
plt.xlabel('Дата')
plt.ylabel('Продажі')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Rolling_Std'], label=f'Ковзне стандартне відхилення (Вікно {win})', color='red', linewidth=2)

plt.title(f'Ковзне стандартне відхилення продажів')
plt.ylabel('Стандартне відхилення')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print(df.head(10).to_string(float_format="%.2f"))

