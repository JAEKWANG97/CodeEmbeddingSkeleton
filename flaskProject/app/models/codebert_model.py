from transformers import RobertaTokenizer, RobertaModel
import torch

# 모델 및 토크나이저 로드
tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
model = RobertaModel.from_pretrained('microsoft/graphcodebert-base')

def get_code_embedding(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding
