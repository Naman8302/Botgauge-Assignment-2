from sentence_transformers import SentenceTransformer
import torch

# Load small model
small_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Load large model
large_model = SentenceTransformer('nvidia/nv-embedqa-e5-v5')

def retrieve_candidates(query, passages, model, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    passage_embeddings = model.encode(passages, convert_to_tensor=True)
    scores = torch.nn.functional.cosine_similarity(query_embedding, passage_embeddings)
    top_k_indices = torch.topk(scores, top_k).indices
    return [passages[i] for i in top_k_indices]