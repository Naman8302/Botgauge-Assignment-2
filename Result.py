import numpy as np

def precision_at_k(retrieved, relevant, k=10):
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)
    true_positives = len(retrieved_set & relevant_set)
    return true_positives / k

def recall_at_k(retrieved, relevant, k=10):
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)
    true_positives = len(retrieved_set & relevant_set)
    return true_positives / len(relevant_set) if relevant_set else 0.0

def dcg(relevances):
    return np.sum((2**relevances - 1) / np.log2(np.arange(1, len(relevances) + 1)))

def ndcg_at_k(retrieved, relevant, k=10):
    # Get the relevance scores for the retrieved documents
    relevances = [1 if doc in relevant else 0 for doc in retrieved[:k]]
    
    # Compute DCG for the retrieved results
    actual_dcg = dcg(relevances)
    
    # Compute IDCG for the ideal results
    ideal_relevances = [1] * min(len(relevant), k) + [0] * (k - min(len(relevant), k))
    ideal_dcg = dcg(sorted(ideal_relevances, reverse=True))
    
    # Handle division by zero
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg

# Example usage:
queries = ["What is the capital of France?", "Who wrote '1984'?"]
retrieved_passages = [
    ["Paris is the capital of France.", "Lyon is a city in France.", "Berlin is the capital of Germany."],
    ["George Orwell wrote '1984'.", "J.K. Rowling wrote 'Harry Potter'.", "Mark Twain wrote 'Tom Sawyer'."]
]
relevant_passages = [
    ["Paris is the capital of France."],
    ["George Orwell wrote '1984'."]
]

# Calculate NDCG@10 for each query
ndcg_scores = [ndcg_at_k(retrieved, relevant) for retrieved, relevant in zip(retrieved_passages, relevant_passages)]
average_ndcg = np.mean(ndcg_scores)

print("NDCG@10 scores for each query:", ndcg_scores)
print("Average NDCG@10:", average_ndcg)

# Sample queries and ground truth relevant passages
queries = ["What is the capital of France?", "Who wrote '1984'?"]
# Assume these are the results from your retrieval stages
retrieved_passages_without_ranking = [
    ["Berlin is the capital of Germany.", "Paris is the capital of France.", "Madrid is the capital of Spain."],
    ["George Orwell wrote '1984'.", "J.K. Rowling wrote 'Harry Potter'.", "Mark Twain wrote 'Tom Sawyer'."]
]

retrieved_passages_with_ranking = [
    ["Paris is the capital of France.", "Lyon is a city in France.", "Berlin is the capital of Germany."],
    ["George Orwell wrote '1984'.", "Aldous Huxley wrote 'Brave New World'.", "J.K. Rowling wrote 'Harry Potter'."]
]

# Calculate NDCG@10 for both retrieval methods
ndcg_without_ranking = [ndcg_at_k(retrieved, relevant) for retrieved, relevant in zip(retrieved_passages_without_ranking, relevant_passages)]
ndcg_with_ranking = [ndcg_at_k(retrieved, relevant) for retrieved, relevant in zip(retrieved_passages_with_ranking, relevant_passages)]

# Calculate averages
average_ndcg_without_ranking = np.mean(ndcg_without_ranking)
average_ndcg_with_ranking = np.mean(ndcg_with_ranking)

# Display results
print("NDCG@10 without ranking models:", ndcg_without_ranking)
print("Average NDCG@10 without ranking models:", average_ndcg_without_ranking)
print("NDCG@10 with ranking models:", ndcg_with_ranking)
print("Average NDCG@10 with ranking models:", average_ndcg_with_ranking)



# Calculate Precision and Recall for both cases
precision_without_ranking = [precision_at_k(retrieved, relevant) for retrieved, relevant in zip(retrieved_passages_without_ranking, relevant_passages)]
recall_without_ranking = [recall_at_k(retrieved, relevant) for retrieved, relevant in zip(retrieved_passages_without_ranking, relevant_passages)]

precision_with_ranking = [precision_at_k(retrieved, relevant) for retrieved, relevant in zip(retrieved_passages_with_ranking, relevant_passages)]
recall_with_ranking = [recall_at_k(retrieved, relevant) for retrieved, relevant in zip(retrieved_passages_with_ranking, relevant_passages)]

print("Precision without ranking:", np.mean(precision_without_ranking))
print("Recall without ranking:", np.mean(recall_without_ranking))
print("Precision with ranking:", np.mean(precision_with_ranking))
print("Recall with ranking:", np.mean(recall_with_ranking))

