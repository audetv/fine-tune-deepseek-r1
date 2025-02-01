# Файл: src/__init__.py

# Импортируем функции для удобства
from .preprocess import load_and_chunk_books, create_dataset, save_dataset
from .train import train_model
from .utils import print_dataset_info, get_device

# Указываем, что будет доступно при импорте *
__all__ = [
    "load_and_chunk_books",
    "create_dataset",
    "save_dataset",
    "train_model",
    "print_dataset_info",
    "get_device",
]