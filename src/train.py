from src.preprocess import load_and_chunk_books, create_dataset, save_dataset
from src.utils import get_device
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk
import torch
import os

def tokenize_dataset(dataset, tokenizer, max_length: int = 512):
    """
    Токенизирует датасет.
    """
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])

def train_model(data_dir: str, output_dir: str, model_name: str = "deepseek-ai/deepseek-r1"):
    """
    Дообучает модель на данных из указанной директории.
    """
    # Выбор устройства (CPU или GPU)
    device = get_device()
    print(f"Используемое устройство: {device}")

    # Загрузка данных
    dataset_path = os.path.join(data_dir, "dataset")
    if os.path.exists(dataset_path):
        print("Загрузка предобработанного датасета...")
        dataset = load_from_disk(dataset_path)
    else:
        print("Предобработанный датасет не найден. Создание нового...")
        texts = load_and_chunk_books(os.path.join(data_dir, "books"))
        dataset = create_dataset(texts)
        save_dataset(dataset, data_dir)

    # Загрузка модели и токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

    # Токенизация данных
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Настройка обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Используем FP16, если доступен GPU
        gradient_accumulation_steps=4,
        warmup_steps=500,
        lr_scheduler_type="linear",
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Создание Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
    )

    # Запуск обучения
    trainer.train()

    # Сохранение модели
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    data_dir = "./data/processed"
    output_dir = "./models/fine-tuned-deepseek-r1"
    train_model(data_dir, output_dir)