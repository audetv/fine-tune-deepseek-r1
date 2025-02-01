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

def save_dataset(dataset: Dataset, output_dir: str, num_examples: int = 5):
    """
    Сохраняет датасет и несколько примеров в текстовый файл.
    """
    # Сохраняем датасет в формате .arrow
    dataset.save_to_disk(os.path.join(output_dir, "dataset"))

    # Сохраняем несколько примеров в текстовый файл
    examples_path = os.path.join(output_dir, "examples.txt")
    with open(examples_path, "w", encoding="utf-8") as f:
        for i in range(min(num_examples, len(dataset))):
            f.write(f"Пример {i + 1}:\n")
            f.write(dataset[i]["text"] + "\n")
            f.write("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    # Пример использования
    books_directory = "./data/books"
    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)

    texts = load_and_chunk_books(books_directory)
    dataset = create_dataset(texts)
    save_dataset(dataset, output_dir)

    print(f"Создан датасет с {len(dataset)} примерами.")
    print(f"Датасет и примеры сохранены в {output_dir}.")