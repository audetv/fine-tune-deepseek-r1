from transformers import BertForMaskedLM, BertTokenizer
import torch

model = BertForMaskedLM.from_pretrained('models/fine-tuned-rubert')
tokenizer = BertTokenizer.from_pretrained('models/fine-tuned-rubert')

text = "Поскольку всякий процесс, поддающийся периодизации, может быть избран в качестве [MASK]."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predicted_token_id = torch.argmax(outputs[0][0, 5])  # индекс [MASK]
predicted_word = tokenizer.decode(predicted_token_id)

print(predicted_word)