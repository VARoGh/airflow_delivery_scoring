# DAG для выполнения ML пайплайна предсказания времени доставки
# Этот DAG демонстрирует типичный production workflow в Airflow:
# 1. Загрузка данных → 2. Новые признаки → 3. Предобработка → 4. Предсказание

# Импорт необходимых модулей Airflow
from airflow import DAG  # Основной класс для создания DAG (Directed Acyclic Graph)
# from airflow.operators.python import PythonOperator  # Оператор для выполнения Python функций
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime  # Для задания временных параметров DAG
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import joblib
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# КОНФИГУРАЦИЯ ПУТЕЙ
# --------------------------------------------------------------------------
# В Airflow пути обычно указываются абсолютные для production окружения
# /opt/airflow - стандартная рабочая директория в Airflow контейнере
BASE_PATH = Path("/opt/airflow")        # Базовая директория проекта в Airflow    # Path.cwd().parent
DATA_PATH = BASE_PATH / "data"          # Директория для данных (входных/выходных)
MODEL_PATH = BASE_PATH / "models"       # Директория для моделей и конфигураций
EMPTY_SCHEMA = ["items_count",
                "distance_km",
                "precip_mm",
                "prep_time_avg",
                "delivery_time_minutes_base",
                "vehicle_type",
                "traffic_level",
                "time_of_day",
                "is_fast_food",
                "is_express_delivery"]          # базовая схема данных
# --------------------------------------------------------------------------
# ФУНКЦИИ ДЛЯ TASKS
# --------------------------------------------------------------------------
# Каждая функция будет выполнена как отдельная задача (task) в Airflow
# Функции должны быть независимыми и идемпотентными (повторный запуск с одними данными дает тот же результат)

def load_data():
    """
    TASK 1: Загрузка сырых данных
    """

    try:
        input_file = DATA_PATH / "input/new_orders_20250220.csv"
        output_file = DATA_PATH / "output/raw.csv"

        if not input_file.exists():
            logger.warning(f"Файл не найден: {input_file}")

            # создаём пустой файл для последующих downstream-задач
            pd.DataFrame(columns=EMPTY_SCHEMA).to_csv(output_file, index=False)
            return None

        # Читаем сырые данные, которые были сохранены при обучении модели
        df = pd.read_csv(input_file)

        logger.info(f"Загружено строк: {len(df)}")

        # Сохраняем для следующего этапа пайплайна
        df.to_csv(DATA_PATH / "output/raw.csv", index=False)

    except Exception as e:
        logger.exception(f"Ошибка в load_data: {e}")

        # создаём пустой файл для последующих downstream-задач
        pd.DataFrame(columns=EMPTY_SCHEMA).to_csv(output_file, index=False)


def feature_engineering():
    """
    TASK 2: Формирование новых кастомных признаков
    """
    output_file = DATA_PATH / "output/feature_engineering.csv"
    try:
        # Читаем сырые данные из предыдущей задачи
        df = pd.read_csv(DATA_PATH / "output/raw.csv")
        logger.info(f"Загружено {len(df)} строк")

        # Проверка файла на пустоту
        if df.empty:
            logger.warning("raw.csv пустой, задача feature engineering пропущена")
            # Создаем пустой файл для последующих downstream-задач
            df.to_csv(output_file, index=False)
            return None

        df['order_datetime'] = pd.to_datetime(df['order_datetime'], errors='coerce')
        bad_rows_datetime = df['order_datetime'].isna().sum()

        if bad_rows_datetime > 0:
            logger.warning(f"Найдены некорректные даты: {bad_rows_datetime}")
        df = df.dropna(subset=['order_datetime'])

        df = df.sort_values(by='order_datetime')

        bad_row_speed = (df['base_speed_kmh'] <= 0).sum()
        if bad_row_speed:
            logger.warning(f"Найдены строки с некорректной скоростью в base_speed_kmh: {bad_row_speed}")

        # Базовое время доставки - distance_km / base_speed_kmh
        df = df[df['base_speed_kmh'] > 0]
        df['delivery_time_minutes_base'] = df['distance_km'] * 60 / df['base_speed_kmh']

        # День недели
        df['day_of_week'] = df['order_datetime'].dt.day_name()

        # Время суток - утро, день, вечер, ночь
        bins = [0, 6, 12, 18, 24]
        labels = ['Ночь', 'Утро', 'День', 'Вечер']
        hours = df['order_datetime'].dt.hour
        df['time_of_day'] = pd.cut(hours, bins=bins, labels=labels, right=False, include_lowest=True)

        # Загружаем конфигурацию предобработки, сохраненную при обучении модели
        cfg = joblib.load(MODEL_PATH / "preprocess_config.pkl")

        # Формируем данные с выбранными признаками
        all_features = cfg['all_features']
        df = df[all_features]

        # Сохраняем предобработанные данные для следующего этапа
        df.to_csv(output_file, index=False)
        logger.info(f"Сохранено {len(df)} строк в файл {output_file}")

    except Exception as e:
        logger.exception(f"Ошибка в feature_engineering: {e}")

        # Создаем пустой файл для последующих downstream-задач
        pd.DataFrame(columns=EMPTY_SCHEMA).to_csv(output_file, index=False)


def preprocess():
    """
    TASK 3: Предобработка данных
    Применяет те же преобразования, что использовались при обучении модели.
    """

    try:
        output_file = DATA_PATH / "output/preprocessed.csv"
        # Читаем сырые данные из предыдущей задачи
        df = pd.read_csv(DATA_PATH / "output/feature_engineering.csv")
        logger.info(f"Загружено {len(df)} строк")

        # Проверка файла на пустоту
        if df.empty:
            logger.warning("feature_engineering.csv пустой, задача preprocess пропущена")
            # Создаем пустой файл для последующих downstream-задач
            df.to_csv(output_file, index=False)
            return None

        # Проверка дубликатов
        dublicated_rows = df.loc[df.duplicated(keep='first')]
        if len(dublicated_rows):
            logger.warning(f"Обнаружены дубликаты: {len(dublicated_rows)}")
            df.drop_duplicates(keep='first', inplace=True)
            logger.info(f"Дубликаты удалены")

        # Загружаем конфигурацию предобработки, сохраненную при обучении модели
        # Это гарантирует, что преобразования будут одинаковыми для обучения и инференса
        cfg = joblib.load(MODEL_PATH / "preprocess_config.pkl")

        # Логарифмическое преобразование c проверкой больше нуля, так как при отр. значении логирифм log1p - RuntimeWarning
        for col in cfg["log_features"]:
            invalid = (df[col] < 0).sum()
            if invalid:
                logger.warning(f"В столбце {col} имеются отрицательные значения: {invalid}")
            df = df[~(df[col] < 0)]
            df[col] = np.log1p(df[col])

        # Сохраняем предобработанные данные для следующего этапа
        df.to_csv(DATA_PATH / "output/preprocessed.csv", index=False)
        logger.info(f"Сохранено {len(df)} строк в файл {output_file}")

    except Exception as e:
        logger.exception(f"Ошибка в preprocess: {e}")

        # Создаем пустой файл для последующих downstream-задач
        pd.DataFrame(columns=EMPTY_SCHEMA).to_csv(output_file, index=False)


def score():
    """
    TASK 4: Выполнение предсказаний (инференс)
    Применяет обученную модель к предобработанным данным.
    В реальных сценариях здесь может быть:
    - Батч-обработка большого объема данных
    - Сохранение результатов в базу данных
    - Отправка уведомлений или результатов
    """
    try:
        # Читаем предобработанные данные
        df = pd.read_csv(DATA_PATH / "output/preprocessed.csv")
        logger.info(f"Загружено {len(df)} строк")

        # Проверка файла на пустоту
        if df.empty:
            logger.warning("preprocessed.csv пустой, задача score не выполнена")
            return None

        # Загружаем обученную модель
        model = joblib.load(MODEL_PATH / "model_cbr_delivery.pkl")

        # Выполняем предсказания
        df["prediction"] = model.predict(df)
        logger.info(f"Сделано предсказаний: {len(df)}")

        # Сохраняем результаты
        df.to_csv(DATA_PATH / "output/new_orders_20250220_predicted.csv", index=False)
        logger.info(f"Сохранено {len(df)} строк в файл {DATA_PATH / "output/new_orders_20250220_predicted.csv"}")
    except Exception as e:
        logger.exception(f"Ошибка в score: {e}")


# --------------------------------------------------------------------------
# ОПРЕДЕЛЕНИЕ DAG
# --------------------------------------------------------------------------
# DAG - это контейнер для задач, определяющий их зависимости и расписание
with DAG(
    dag_id="delivery_production_scoring",  # Уникальный идентификатор DAG
    start_date=datetime(2024, 1, 1),          # Дата начала работы DAG
    schedule="*/2 * * * *",                  # Расписание (None - ручной запуск)
    # schedule="* * * * *",                   # Каждую минуту
    catchup=False,                            # Не выполнять пропущенные запуски
    tags=["ml", "catboost"],          # Теги для поиска в UI
) as dag:
    # Внутри этого блока определяем задачи и их зависимости

    # ----------------------------------------------------------------------
    # ОПРЕДЕЛЕНИЕ ЗАДАЧ (TASKS)
    # ----------------------------------------------------------------------
    # Каждая задача - это экземпляр оператора (в данном случае PythonOperator)

    load_task = PythonOperator(
        task_id="load_data",          # Уникальный ID задачи в рамках DAG
        python_callable=load_data,    # Функция для выполнения
    )

    feature_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess,
    )

    score_task = PythonOperator(
        task_id="score",
        python_callable=score,
    )

    # ----------------------------------------------------------------------
    # ОПРЕДЕЛЕНИЕ ЗАВИСИМОСТЕЙ МЕЖДУ ЗАДАЧАМИ
    # ----------------------------------------------------------------------
    load_task >> feature_task >> preprocess_task >> score_task

    # Альтернативная запись с явным указанием:
    # load_task.set_downstream(feature_task)
    # feature_task.set_downstream(preprocess_task)
    # preprocess_task.set_downstream(score_task)

# Тест
# load_data()
# feature_engineering()
# preprocess()
# score()
