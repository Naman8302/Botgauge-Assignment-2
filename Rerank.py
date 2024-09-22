from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load ranking models
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
cross_encoder_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')

def rerank_passages(query, passages):
    inputs = cross_encoder_tokenizer([(query, passage) for passage in passages], return_tensors='pt', padding=True)
    outputs = cross_encoder_model(**inputs)
    scores = outputs.logits.softmax(dim=1)[:, 1]  # Assuming binary classification (relevant / not relevant)
    sorted_indices = scores.argsort(descending=True)
    return [passages[i] for i in sorted_indices]