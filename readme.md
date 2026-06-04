# 🏠 Delivery Prediction ML Pipeline

**Производственный ML-пайплайн для батч-скоринга времени доставки с использованием Apache Airflow**

## 🚀 Быстрый старт

> ⚠️ Требования:
>
> * Docker + Docker Compose
> * Python 3.9+
> * Catboost 1.2.5+
> * Git

---

## 🔧 1. Подготовка окружения

### 🐧 Linux / 🍎 macOS

```bash
# 1. Создаём .env файл с UID пользователя (нужно для прав в Docker)
echo "AIRFLOW_UID=$(id -u)" > .env

# 2. Создаём виртуальное окружение
python3 -m venv venv

# 3. Активируем окружение
source venv/bin/activate

# 4. Устанавливаем зависимости
pip install -r requirements.txt
```

---

### 🪟 Windows (PowerShell)

```powershell
# 1. Создаём .env файл (AIRFLOW_UID не нужен на Windows)
# Использовать произвольный ID для совместимости
"AIRFLOW_UID=50000" | Out-File -FilePath .env -Encoding ASCII
```

> 📌 **Важно для Windows:**
> `AIRFLOW_UID` используется для Linux/macOS, на Windows Docker Desktop сам управляет правами.

```powershell
# 2. Создаём виртуальное окружение
python -m venv venv

# 3. Активируем окружение
venv\Scripts\Activate.ps1

# 4. Устанавливаем зависимости
pip install -r requirements.txt
```

> ❗ Если PowerShell ругается на политику выполнения:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 🐳 2. Запуск Airflow для батч-скоринга

### Инициализация Airflow (один раз)

```bash
docker compose up airflow-init --build
```

### Запуск всех сервисов

```bash
docker compose up -d
```

### Проверка статуса

```bash
docker compose ps
```

---

## 🌐 3. Работа с Airflow UI

* **Airflow UI**: [http://localhost:8080](http://localhost:8080)
* **Логин**: `airflow`
* **Пароль**: `airflow`
* **DAG**: `delivery_scoring`

### 🎯 Как запустить батч-скоринг:

1. Откройте Airflow UI
2. Найдите DAG `delivery_scoring`
3. Нажмите ▶️ кнопку "Trigger DAG"
4. Следите за выполнением в Graph View

---

## 📁 Структура проекта: обучение Airflow на примере

```
airflow-docker/
├── dags/                    # Airflow DAG'и - СЕРДЦЕ ПРОЕКТА
│   └── delivery_scoring.py  # Основной DAG для батч-скоринга
├── data/                     # Данные (volume для Docker)
│   ├── input/               # Новые данные для скоринга
│   └── output/              # Результаты предсказаний
├── models/                   # ML модели (обучаются локально)
├── logs/                     # Логи Airflow для отладки
├── plugins/                  # Кастомные плагины Airflow
├── config/                   # Конфигурационные файлы
├── delivery_model.py        # Локальное обучение модели
├── docker-compose.yaml      # Конфигурация всех сервисов
└── README.md
```

---

## 🔄 ML пайплайн в Airflow

**Этот DAG демонстрирует типичный батч-скоринг:**

```python
## Граф выполнения в Airflow:
load_data → feature_engineering → preprocess → score → [save_results]
```
1. **`load_data`** — Airflow Task: загрузка новых данных для предсказания
2. **`feature_engineering`** - Airflow Task: формирование новых признаков
2. **`preprocess`** — Airflow Task: применение сохранённого препроцессинга
3. **`score`** — Airflow Task: инференс модели
4. **`save_results`** — сохранение предсказаний (демонстрация)

---

## 🐳 Полезные команды

```bash
# Зайти внутрь контейнера Airflow (для понимания структуры)
docker compose exec airflow-scheduler bash
ls /opt/airflow/

# Посмотреть логи выполнения DAG
docker compose logs -f airflow-scheduler

# Запустить DAG вручную (из консоли)
docker compose exec airflow-scheduler airflow dags trigger house_price_production_scoring

# Остановить все сервисы
docker compose down

# Полная очистка (осторожно!)
docker compose down -v
```