import torch
import torch.nn as nn
import pickle
import streamlit as st

# ==========================================
# 1️⃣ LOAD METADATA
# ==========================================
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

word_to_idx = metadata["word_to_idx"]
idx_to_word = metadata["idx_to_word"]
max_len = metadata["max_len"]
vocab_size = metadata["vocab_size"]

# ==========================================
# 2️⃣ MODEL ARCHITECTURE (MATCH TRAINING)
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

# ==========================================
# 3️⃣ LOAD MODEL
# ==========================================
EMBED_DIM = 100
HIDDEN_DIM = 150

model = LSTMModel(vocab_size, EMBED_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# ==========================================
# 4️⃣ PREDICTION FUNCTION
# ==========================================
def predict_next_word(seed_text):
    words = seed_text.lower().split()
    tokens = [word_to_idx[w] for w in words if w in word_to_idx]

    if not tokens:
        return "No known words found."

    padded_tokens = [0] * (max_len - 1 - len(tokens)) + tokens
    input_tensor = torch.tensor([padded_tokens], dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()

    return idx_to_word[predicted_idx]

# ==========================================
# 5️⃣ STREAMLIT UI
# ==========================================
st.title("🔮 LSTM Next Word Prediction")
st.write("Enter a sentence and the model will predict the next word.")

user_input = st.text_input("Enter text:")

if st.button("Predict"):
    if user_input.strip() != "":
        next_word = predict_next_word(user_input)
        st.success(f"Predicted Next Word: {next_word}")
    else:
        st.warning("Please enter some text.")
