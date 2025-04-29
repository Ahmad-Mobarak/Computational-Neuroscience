import numpy as np

# Training data
sentences = [
    ("neural networks are cool", "cool"),
    ("ai is the future", "future"),
    ("deep learning rocks", "rocks"),
]

# Make words into numbers
def get_words(text):
    return text.lower().split()

# Get all unique words
unique_words = set()
for text, _ in sentences:
    unique_words.update(get_words(text))

# Make word to number dictionary
word_to_num = {word: i for i, word in enumerate(sorted(unique_words))}
num_to_word = {i: word for word, i in word_to_num.items()}
total_words = len(word_to_num)

# Set random seed for reproducibility
np.random.seed(42)

# Model settings
word_vec_size = 4
hidden_size = 4
learning_rate = 0.01

# Initialize weights
word_vectors = np.random.randn(total_words, word_vec_size)
forward_W1 = np.random.randn(hidden_size, word_vec_size)
forward_W2 = np.random.randn(hidden_size, hidden_size)
output_W = np.random.randn(total_words, hidden_size * 2)
backward_W1 = np.random.randn(hidden_size, word_vec_size)
backward_W2 = np.random.randn(hidden_size, hidden_size)

def get_probs(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def get_loss(pred, true):
    return -np.log(pred[true] + 1e-9)

# Training loop
for step in range(500):
    total_error = 0
    
    for text, target in sentences:
        # Convert words to numbers
        word_nums = [word_to_num[w] for w in get_words(text)]
        input_seq, target_num = word_nums[:3], word_to_num[target]

        # Forward pass
        forward_states = []
        forward_h = np.zeros((hidden_size,))
        for num in input_seq:
            vec = word_vectors[num]
            forward_h = np.tanh(forward_W1 @ vec + forward_W2 @ forward_h)
            forward_states.append((forward_h.copy(), vec))

        # Backward pass
        backward_states = []
        backward_h = np.zeros((hidden_size,))
        for num in reversed(input_seq):
            vec = word_vectors[num]
            backward_h = np.tanh(backward_W1 @ vec + backward_W2 @ backward_h)
            backward_states.insert(0, (backward_h.copy(), vec))

        # Combine forward and backward
        combined_h = np.concatenate((forward_h, backward_h))
        scores = output_W @ combined_h
        probs = get_probs(scores)
        error = get_loss(probs, target_num)
        total_error += error

        # Calculate gradients
        d_scores = probs
        d_scores[target_num] -= 1
        d_output_W = np.outer(d_scores, combined_h)

        # Backprop forward
        d_forward_W1 = np.zeros_like(forward_W1)
        d_forward_W2 = np.zeros_like(forward_W2)
        d_forward_h = np.zeros((hidden_size,))
        for t in reversed(range(len(input_seq))):
            h, vec = forward_states[t]
            d_tanh = (1 - h ** 2) * (d_output_W[:, :hidden_size].sum() + d_forward_h)
            d_forward_W1 += np.outer(d_tanh, vec)
            d_forward_W2 += np.outer(d_tanh, forward_states[t-1][0] if t > 0 else np.zeros_like(h))
            d_forward_h = forward_W2.T @ d_tanh
            word_vectors[input_seq[t]] -= learning_rate * (forward_W1.T @ d_tanh)

        # Backprop backward
        d_backward_W1 = np.zeros_like(backward_W1)
        d_backward_W2 = np.zeros_like(backward_W2)
        d_backward_h = np.zeros((hidden_size,))
        for t in range(len(input_seq)):
            h_b, vec = backward_states[t]
            d_tanh_b = (1 - h_b ** 2) * (d_output_W[:, hidden_size:].sum() + d_backward_h)
            d_backward_W1 += np.outer(d_tanh_b, vec)
            d_backward_W2 += np.outer(d_tanh_b, backward_states[t-1][0] if t > 0 else np.zeros_like(h_b))
            d_backward_h = backward_W2.T @ d_tanh_b
            word_vectors[input_seq[t]] -= learning_rate * (backward_W1.T @ d_tanh_b)

        # Update weights
        output_W -= learning_rate * d_output_W
        forward_W1 -= learning_rate * d_forward_W1
        forward_W2 -= learning_rate * d_forward_W2
        backward_W1 -= learning_rate * d_backward_W1
        backward_W2 -= learning_rate * d_backward_W2

    if step % 100 == 0:
        print(f"Step {step}, Error: {total_error:.4f}")

# Test the model
print("\nTesting the model ---")
test_text = "neural networks are"
test_nums = [word_to_num[w] for w in get_words(test_text)]

# Forward pass
forward_h = np.zeros((hidden_size,))
print("\nForward states:")
for num in test_nums:
    vec = word_vectors[num]
    forward_h = np.tanh(forward_W1 @ vec + forward_W2 @ forward_h)
    print(f"Word: {num_to_word[num]}, State: {forward_h}")

# Backward pass
backward_h = np.zeros((hidden_size,))
print("\nBackward states:")
for num in reversed(test_nums):
    vec = word_vectors[num]
    backward_h = np.tanh(backward_W1 @ vec + backward_W2 @ backward_h)
    print(f"Word: {num_to_word[num]}, State: {backward_h}")

# Make prediction
combined_h = np.concatenate((forward_h, backward_h))
scores = output_W @ combined_h
probs = get_probs(scores)
predicted_num = np.argmax(probs)
print("\nPredicted word:", num_to_word[predicted_num]) 