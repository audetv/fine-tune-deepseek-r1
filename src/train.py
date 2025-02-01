from datasets import load_from_disk, Dataset
import torch
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from src import save_dataset, load_and_chunk_books, create_dataset

def tokenize_dataset(dataset, tokenizer, max_length: int = 512):
    """
    Токенизирует датасет.
    """
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])

def train_model(data_dir: str, output_dir: str, model_name: str = "sberbank-ai/ruBert-base"):
    """
    Дообучает модель на данных из указанной директории.
    """
    # Выбор устройства (CPU или GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    # Токенизация данных
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Разделение данных на обучающую и тестовую выборки
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Создание DataCollator для задачи MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,  # Включаем маскирование
        mlm_probability=0.15  # Вероятность маскирования токенов
    )

    # Настройка обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=500,
        eval_strategy="steps",  # Оценка каждые eval_steps шагов
        eval_steps=500,  # Частота оценки
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Используем FP16, если доступен GPU
        gradient_accumulation_steps=4,
        warmup_steps=500,
        lr_scheduler_type="linear",
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Создание Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Передаём набор данных для оценки
        data_collator=data_collator,
    )

    # Запуск обучения
    trainer.train()

    # Сохранение модели
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    data_dir = "./data/processed"
    output_dir = "./models/fine-tuned-rubert"
    train_model(data_dir, output_dir)