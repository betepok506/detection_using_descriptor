# Описание
TO do
# Запуск

Для установки всех зависимостей воспользуйтесь командой:
```commandline
pip install -r requirements.txt
```

# Train
Если в `configs/train_config.yaml` параметр `use_mlflow=True` то перед запуском необходимо установить следующие переменные окружения:
```commandline
AWS_ACCESS_KEY_ID=admin
AWS_SECRET_ACCESS_KEY=sample_key
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
MLFLOW_TRACKING_URI=http://localhost:5000
```

## MLflow
Для установки MLflow можно воспользоваться следующим репозиторием:

```commandline
https://github.com/Toumash/mlflow-docker
```

## Конфигурирование
Конфигурирование проекта осуществляется путем изменения `configs/train_config.yaml`
соответствующих параметров:

```commandline
Тут будет описание параметров
```
# Тестирование

# Оценка 

Оценка осуществляется с помощью 
```commandline
https://github.com/rafaelpadilla/Object-Detection-Metrics
```


# Результаты
TODO
