from pipeline import CountryClusteringPipeline

def start_cli():
    pipeline = CountryClusteringPipeline()
    print("--- Предсказание кластера страны ---")
    print("Пожалуйста, вводите показатели по очереди (только цифры):")

    # Словарь с понятными описаниями на русском
    feature_prompts = {
        'child_mort': 'Смертность детей до 5 лет (на 1000 рожденных)',
        'exports': 'Экспорт товаров и услуг (% от ВВП)',
        'health': 'Расходы на здравоохранение (% от ВВП)',
        'imports': 'Импорт товаров и услуг (% от ВВП)',
        'income': 'Чистый доход на душу населения',
        'inflation': 'Инфляция (годовой %)',
        'life_expec': 'Средняя продолжительность жизни (лет)',
        'total_fer': 'Коэффициент рождаемости (количество детей на женщину)',
        'gdpp': 'ВВП на душу населения'
    }

    user_data = {}
    for feature, prompt in feature_prompts.items():
        while True:
            val = input(f"{prompt}: ").strip()
            try:
                user_data[feature] = float(val)
                break  # Успешный ввод, переходим к следующему вопросу
            except ValueError:
                print("Ошибка: введите цифру!")

    cluster = pipeline.predict_new(user_data)

    cluster_names = {
        0: "Развитая экономика",
        1: "Страна под риском / Бедная",
        2: "Страна со средним достатком"
    }
    print(f"\nРезультат: Страна относится к КЛАСТЕРУ {cluster}")
    print(f"Описание: {cluster_names.get(cluster, 'Неизвестно')}")

if __name__ == "__main__":
    start_cli()
