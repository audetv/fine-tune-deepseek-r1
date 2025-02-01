def print_dataset_info(dataset):
    """
    Выводит информацию о датасете.
    """
    print(f"Размер датасета: {len(dataset)}")
    print(f"Пример данных: {dataset[0]}")

import torch

def get_device():
    """
    Возвращает доступное устройство (CPU или GPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")