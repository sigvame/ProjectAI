import matplotlib.pyplot as plt
import numpy as np


#завдання 1

# months = ["Сiч", "Лют", "Бер", "Квіт", "Тр", "Чер", "Лип", "Сер", "Вер", "Жов", "Лис", "Гр"]
# plan = np.array([100, 120, 150, 140, 170, 180, 200, 195, 230, 150, 150, 180])
# fact = np.array([95, 125, 145, 160, 165, 190, 210, 160, 215, 180, 110, 200])
#
# plt.figure(figsize=(10, 6))
#
# plt.plot(months, plan, label="План", marker='o', linestyle='--', color='blue')
# plt.plot(months, fact, label="Факт", marker='x', linestyle='-', color='green')
#
# plt.xlabel("\nМісяць")
# plt.ylabel("Сума продажів у Тисячах грн\n")
# plt.title("Графік планових та фактичних продажів")
#
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.legend()
#
# plt.show()



#завдання 2

# np.random.seed(42)
# ages = np.random.randint(10, 90, 100)
#
# plt.figure(figsize=(10, 6))
# plt.hist(ages, bins=10, edgecolor='black', alpha=0.7, color='lightblue')
#
# av_age = np.mean(ages)
# plt.axvline(av_age, color='red', linestyle='dashed', linewidth=2, label=f'Середній вік: {av_age:.2f}')
#
# plt.xlabel("Вік")
# plt.ylabel("Кількість осіб")
# plt.title("Розподіл віку 100 осіб")
#
# plt.legend()
# plt.grid(axis='y', linestyle=':', alpha=0.7)
#
# plt.show()


#завдання 3

# np.random.seed(42)
#
# group_a = np.random.normal(loc=90, scale=5, size=50)
# group_b = np.random.normal(loc=70, scale=12, size=50)
# group_c = np.random.normal(loc=80, scale=8, size=50)
#
# data = [group_a, group_b, group_c]
# group_labels = ['Група А', 'Група Б', 'Група В']
#
# plt.figure(figsize=(10, 7))
#
# plt.boxplot(data, tick_labels=group_labels, patch_artist=True, medianprops={'color': 'red'})
#
# plt.xlabel("\nГрупа студентів")
# plt.ylabel("Оцінка іспиту")
# plt.title("Порівняння розподілу оцінок іспиту між групами")
#
# plt.ylim(50, 105)
# plt.grid(axis='y', linestyle=':', alpha=0.7)
#
# plt.show()




#завдання 4

# dates = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
# temperature = [20, 22, 25, 23, 21, 18, 19]
# w = [10, 15, 20, 10, 40, 30, 50] #вологість
#
# fig, l1 = plt.subplots(figsize=(10, 6))
#
# l1.set_ylabel('Температура (°C)', color='red')
# l1.plot(dates, temperature, color='red', label='Температура')
# l1.tick_params(axis='y', labelcolor='red')
#
# l2 = l1.twinx()
# l2.set_ylabel('Вологість (%)', color='blue')
# l2.plot(dates, w, color='blue', linestyle='--', label='Вологість')
# l2.tick_params(axis='y', labelcolor='blue')
#
# l1.set_xlabel("День тижня")
# plt.title("Денні значення температури та вологості")
# l1.tick_params(axis='x', rotation=30)
#
# lines = l1.get_lines() + l2.get_lines()
# labels = [l.get_label() for l in lines]
# l1.legend(lines, labels, loc='upper right')
#
# fig.tight_layout()
# plt.show()



#завдання 5

hours = np.arange(24)

np.random.seed(42)
base_load = np.sin(hours / 24 * 2 * np.pi - np.pi / 2) * 50 + 70
load_data = base_load + np.random.normal(0, 5, 24)
load_data = np.maximum(load_data, 20)

plt.figure(figsize=(12, 6))
plt.plot(hours, load_data, color='darkblue', linewidth=2, label="Навантаження сервера")
plt.fill_between(hours, load_data, color='lightblue', alpha=0.5)

plt.xlabel("Година доби (0-23)")
plt.ylabel("Навантаження (умовні одиниці)")
plt.title("Погодинне навантаження сервера за добу")

plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(hours[::2])
plt.xlim(0, 23)

plt.legend(loc='upper left')
plt.show()
