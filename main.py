import pandas as pd



# завдання 1

data = {
    "Назва": ["Дюна", "Атлас розправляє плечі", "Гаррі Поттер",
              "Володар перснів", "1984", "Автостопом по галактиці"],
    "Автор": ["Френк Герберт", "Айн Ренд", "Джоан Роулінг",
              "Дж. Р. Р. Толкін", "Джордж Орвелл", "Дуглас Адамс"],
    "Рік видання": [1965, 1957, 1997, 1954, 1949, 1979],
    "Ціна": [350, 480, 400, 520, 290, 310]
}

df = pd.DataFrame(data)

print("DataFrame:")
print(df)
print("-" * 30)

average_price = df["Ціна"].mean()
print(f"Середня ціна книг: {average_price:.2f} грн")
print("-" * 30)

r_books = df[df["Рік видання"] > 2015]
print("Книги, видані після 2015 року:")
if not r_books.empty:
    print(r_books)
else:
    print("Немає книг після 2015 року")
print("-" * 30)

sorted_data = df.sort_values(by="Ціна", ascending=True)
print("")
print(sorted_data)





#завдання 2

# df = pd.read_csv('orders.csv')
#
# print("Перші 10 рядків даних:")
# print(df.head(10))
# print("\n" + "=" * 50 + "\n")
#
# orders_by_client = df['Клієнт'].value_counts()
# print("Кількість замовлень кожного клієнта:")
# print(orders_by_client)
# print("\n" + "=" * 50 + "\n")
#
# max_sum = df['Сума'].max()
# min_sum = df['Сума'].min()
# print(f"Максимальна сума замовлення: {max_sum}")
# print(f"Мінімальна сума замовлення: {min_sum}")
# print("\n" + "=" * 50 + "\n")
#
# total_sum = df['Сума'].sum()
# print(f"Загальна сума всіх замовлень: {total_sum}")


#завдання 3

# data = {
#     "Продукт": ["Мигдаль", "Яблуко", "Куряче філе", "Рис", "Вівсянка",
#                 "Авокадо", "Банан", "Стейк", "Броколі", "Сир твердий"],
#     "Категорія": ["Горіхи", "Фрукти", "М'ясо", "Зернові", "Зернові",
#                   "Фрукти", "Фрукти", "М'ясо", "Овочі", "Молочні"],
#     "Калорії": [576, 52, 165, 130, 389,
#                 160, 89, 271, 55, 402],
#     "Білки": [21, 0.3, 31, 2, 16,
#               2, 1, 26, 3, 25]
# }
#
# df = pd.DataFrame(data)
#
# print("DataFrame:")
# print(df)
# print("-" * 50)
#
# products = df[df["Калорії"] > 300]
# print("Продукти з калорійністю вище 300:")
# print(products)
# print("-" * 50)
#
# protein = df.groupby("Категорія")["Білки"].mean()
# print("Середня кількість білків за категоріями:")
# print(protein)
# print("-" * 50)
#
# sorted_data = df.sort_values(by="Калорії", ascending=False)
# print(sorted_data)




#завдання 4

# data = {
#     "Ім'я": ["Іван", "Микита", "Андрій", "Вікторія", "Микита", "Іван", "Андрій", "Катерина", "Катерина"],
#     "Проект": ["Проєкт А", "Проєкт Б", "Проєкт В", "Проєкт Б", "Проєкт А", "Проєкт В", "Проєкт А", "Проєкт Б", "Проєкт В"],
#     "Години": [15, 20, 10, 25, 30, 10, 5, 20, 15]
# }
#
# df = pd.DataFrame(data)
#
# print("DataFrame:")
# print(df)
# print("-" * 50)
#
# employee_hours = df.groupby("Ім'я")["Години"].sum()
# print("Загальна кількість годин за кожним співробітником:")
# print(employee_hours)
# print("-" * 50)
#
# project_hours = df.groupby("Проект")["Години"].sum()
# print("Загальна кількість годин за кожним проектом:")
# print(project_hours)
# print("-" * 50)
#
# employee_name = employee_hours.idxmax()
# employee_h = employee_hours.max()
# print(f"Співробітник, який витратив найбільше годин: {employee_name}")
# print(f"Кількість годин: {employee_h}")