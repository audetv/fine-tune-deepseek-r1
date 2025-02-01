# Fine-Tune DeepSeek-R1

Этот проект предназначен для дообучения модели DeepSeek-R1 на пользовательских данных (книгах в формате .txt).

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/audetv/fine-tune-deepseek-r1.git
   cd fine-tune-deepseek-r1

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Поместите книги в папку data/books

## Запуск

1. Предобработка данных:

```bash
python src/preprocess.py
```

2. Обучение модели:

```bash
python src/train.py
```

## Структура проекта
- `data/books/`: Папка с книгами в формате .txt.
- `src/`: Исходный код.
- `models/`: Сохраненные модели.
- `logs/`: Логи обучения.

```
fine-tune-deepseek-r1/
├── data/
│   └── books/               # Папка с вашими книгами в формате .txt
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Скрипт для предобработки данных
│   ├── train.py             # Скрипт для обучения модели
│   └── utils.py             # Вспомогательные функции
├── logs/                    # Логи обучения (создастся автоматически)
├── models/                  # Сохраненные модели (создастся автоматически)
├── requirements.txt         # Файл с зависимостями
└── README.md                # Описание проекта
```

---

### 6. Запуск проекта
1. Установите зависимости:

```bash
pip install -r requirements.txt
```
2. Поместите книги в папку `data/books`.
2. Запустите предобработку данных:

```bash
python src/preprocess.py
```

Запустите обучение:
```bash
python src/train.py
```
