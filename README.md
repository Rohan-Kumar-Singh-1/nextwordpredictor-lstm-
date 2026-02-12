Next-Word Predictor using LSTM

This project is a deep learning model built with **PyTorch** that predicts the next word in a sequence. It was built following the CampusX PyTorch tutorial series.

Project Overview
The model uses a Long Short-Term Memory (LSTM) network to learn patterns from a technical text dataset. Given a sequence of words, it calculates the probability of the next word in the vocabulary.

Model Architecture
* **Embedding Layer**: Converts word tokens into 100-dimensional dense vectors.
* **LSTM Layer**: Processes the sequence (150 hidden units) to capture context.
* **Fully Connected Layer**: Maps the output to the total vocabulary size for prediction.

How to Run
1. Open the `NextWordPredictor.ipynb` file in **Google Colab**.
2. Run all cells to train the model on the provided dataset.
3. Use the `predict_next_words` function at the bottom to test your own sentences.

Sample Output
* Input: "Machine learning is" 
* Predicted: "a subset of artificial intelligence"
