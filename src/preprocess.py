import os
from datasets import Dataset

def load_and_chunk_books(directory: str, chunk_size: int = 512) -> list:
    """
    Загружает тексты из файлов .txt и разбивает их на фрагменты.
    """
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                text = f.read()
                # Разделяем текст на chunks фиксированной длины
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    texts.append(chunk)
    return texts

def create_dataset(texts: list) -> Dataset:
    """
    Создает датасет из списка текстов.
    """
    return Dataset.from_dict({"text": texts})

if __name__ == "__main__":
    # Пример использования
    books_directory = "./data/books"
    texts = load_and_chunk_books(books_directory)
    dataset = create_dataset(texts)
    print(f"Создан датасет с {len(dataset)} примерами.")