Great question: if all word embeddings or tokens are represented as floating point vectors, how do you find the next word number (i.e., the next word in a sequence)?



There are two interpretations of your question — I’ll address both clearly:

✅ Interpretation 1: “How do I map a word embedding to the next word?”


This is what a language model learns to do — it predicts the next word given context.



💡 Process in Transformer-based models:
Input: "The sky is"

Tokenized: [101, 1996, 3712, 2003, 102]

Embeddings: Floating-point vectors for each token.

Model output: Predicts next token using softmax over the vocabulary.

# Example with transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

input_ids = tokenizer("The sky is", return_tensors="pt").input_ids
output = model(input_ids)
logits = output.logits

# Get prediction for next token (last position)
next_token_logits = logits[0, -1]
next_token_id = torch.argmax(next_token_logits)
next_word = tokenizer.decode(next_token_id)

print(next_word)  # e.g., "blue"
So: even though the model uses floating point under the hood, it outputs the index of the next token, which you can decode to the next word.

✅ Interpretation 2: “I have a word’s embedding — how do I get its word index or string?”


This is a reverse lookup of an embedding — not straightforward, but here’s the idea:



🔁 Option 1: Compare to embedding matrix
# Assume you have an embedding matrix: vocab_size x embedding_dim
embedding_matrix = model.transformer.wte.weight  # GPT-2 example
input_embedding = some_vector  # shape: [embedding_dim]

# Compute cosine similarity to all embeddings
similarities = torch.nn.functional.cosine_similarity(
    embedding_matrix, input_embedding.unsqueeze(0), dim=1
)
top_id = torch.argmax(similarities)
decoded_word = tokenizer.decode(top_id)
This gives the closest word to a floating point vector — useful for visualization or probing.

🧠 Summary Table
Task

Goal

How

Predict next word

Given context, get next token

Model forward pass → softmax over vocab

Reverse from embedding

Floating vector → token

Cosine similarity vs embedding matrix

Use “word number”

Get index of token

tokenizer.convert_tokens_to_ids()

Decode “word number”

Index → token/word

tokenizer.decode()

Let me know whether you’re building your own embedding model, using a pretrained one, or doing something low-level — I can show code accordingly.



