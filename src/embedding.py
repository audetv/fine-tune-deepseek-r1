from transformers import AutoTokenizer, AutoModel
import torch

def get_embeddings(texts, model_path, pooling="cls"):
    """
    Извлекает эмбеддинги для списка текстов.
    """
    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    # Токенизация текстов
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Получение эмбеддингов
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

        # Выбор способа пулинга
        if pooling == "cls":
            embeddings = last_hidden_state[:, 0, :]  # Берём эмбеддинг токена [CLS]
        elif pooling == "mean":
            embeddings = torch.mean(last_hidden_state, dim=1)  # Усреднение по всем токенам
        else:
            raise ValueError("Неизвестный тип пулинга. Используйте 'cls' или 'mean'.")

    return embeddings.numpy()

# Пример использования
model_path = "./models/fine-tuned-rubert"  # Путь к сохранённой модели
texts = ["ДОТУ", "ПФУ"]  # Ваши тексты
embeddings = get_embeddings(texts, model_path, pooling="cls")
print("Эмбеддинги:", embeddings)