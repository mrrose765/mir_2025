def precision_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    true_positives = sum([1 for item in retrieved_at_k if item in relevant])
    return true_positives / k

def recall(retrieved, relevant):
    true_positives = sum([1 for item in retrieved if item in relevant])
    return true_positives / len(relevant) if relevant else 0

def average_precision(retrieved, relevant):
    ap = 0.0
    hit_count = 0
    for i, item in enumerate(retrieved):
        if item in relevant:
            hit_count += 1
            ap += hit_count / (i + 1)
    return ap / len(relevant) if relevant else 0

def r_precision(retrieved, relevant):
    R = len(relevant)
    retrieved_at_r = retrieved[:R]
    true_positives = sum([1 for item in retrieved_at_r if item in relevant])
    return true_positives / R if R > 0 else 0
