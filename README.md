# Next Word Prediction using LSTM

This project implements a sequence-to-word generative model using Long Short-Term Memory (LSTM) networks. Unlike standard neural networks, this model can process entire sequences of text and maintain a "memory" of earlier words to predict what comes next.

## Overview

Predicting the next word is a fundamental task in NLP, powering features like autocomplete and predictive text. This project uses a **Recurrent Neural Network (RNN)** variant called **LSTM**, which is specifically designed to solve the "vanishing gradient" problem, allowing the model to remember long-term dependencies in text.

## Dataset

- **Source:** 2024 State of the Union Address (`2024_state_of_the_union.txt`).
- **Nature:** A long-form political speech containing complex sentence structures and specific vocabulary related to governance, economy, and foreign policy.

## Objectives

- **Text Tokenization:** Converting raw text into a stream of integers where each unique word is assigned an index.
- **N-gram Generation:** Creating training sequences by sliding a window across the text (e.g., "I am" -> "learning", "I am learning" -> "AI").
- **Sequence Padding:** Ensuring all input sequences are of the same length to be processed by the GPU.
- **Categorical Prediction:** Treating the next word prediction as a multi-class classification problem where the number of classes equals the vocabulary size.

## The LSTM Architecture

The model is built using the Keras Sequential API:
1. **Embedding Layer:** Learns dense vector representations of words, capturing semantic relationships.
2. **LSTM Layer:** The "brain" of the model, processing the sequence and maintaining a hidden state.
3. **Dense Output Layer:** A Softmax layer that calculates the probability distribution across all words in the vocabulary.



## Methods and Analysis

- **Data Preparation:** - Tokenized the entire speech.
  - Generated input-output pairs where the input is a sequence of words and the output is the immediate next word.
- **Training:** - The model learns to minimize `categorical_crossentropy` loss.
  - Used the Adam optimizer for efficient gradient descent.
- **Inference (The "Creative" Part):**
  - Given a seed phrase (e.g., "We must"), the model predicts the next word, appends it to the phrase, and repeats the process to generate a full sentence.



## Tech Stack

- **Language:** Python 3
- **Deep Learning:** TensorFlow / Keras
- **Numerical Processing:** NumPy
- **Environment:** Jupyter / Google Colab (GPU accelerated)

## How to Run

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/next-word-prediction-lstm.git
   cd next-word-prediction-lstm

2. *Install dependencies:*
   pip install tensorflow numpy

3. *Open the notebook:*
   jupyter notebook 36_LSTM_next_word_prediction.ipynb

4. Provide Text Source: Ensure the text file 2024_state_of_the_union.txt is in the directory.
