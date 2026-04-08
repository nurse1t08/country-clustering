import pandas as pd
from pipeline import CountryClusteringPipeline
import os

# Все пути теперь ведут прямо в папку archive
DATA_PATH = "/Users/aizat/Downloads/archive/Country-data.csv"
PROJECT_ROOT = "/Users/aizat/Downloads/archive/"

def run_training_pipeline():
    if not os.path.exists(DATA_PATH):
        print(f"Ошибка: Файл {DATA_PATH} не найден! Проверьте наличие Country-data.csv.")
        return

    print("Запуск обучения...")
    df = pd.read_csv(DATA_PATH)
    
    pipeline = CountryClusteringPipeline(n_clusters=3)
    df_clustered = pipeline.train(df)

    pipeline.save_models(os.path.join(PROJECT_ROOT, "models"))
    df_clustered.to_csv(os.path.join(PROJECT_ROOT, "clustered_countries.csv"), index=False)
    print("Обучение завершено. Файл clustered_countries.csv создан.")

if __name__ == "__main__":
    run_training_pipeline()
