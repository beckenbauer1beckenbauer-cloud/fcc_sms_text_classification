import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

# 1. Download and read datasets (Paths provided by freeCodeCamp env)
train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

df_train = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
df_test = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

# 2. Map text labels to numbers
df_train['label'] = df_train['label'].map({'ham': 0, 'spam': 1})
df_test['label'] = df_test['label'].map({'ham': 0, 'spam': 1})

# 3. Text Preprocessing (Tokenization & Padding)
max_words = 1000
max_len = 50

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_train['message'])

train_sequences = tokenizer.texts_to_sequences(df_train['message'])
train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')

test_sequences = tokenizer.texts_to_sequences(df_test['message'])
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')

# 4. Neural Network Architecture
model = tf.keras.Sequential([
    layers.Embedding(input_dim=max_words, output_dim=16, input_length=max_len),
    layers.GlobalAveragePooling1D(),
    layers.Dense(24, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train Model
model.fit(train_padded, df_train['label'], epochs=10, validation_data=(test_padded, df_test['label']), verbose=0)

# 6. The Mandatory freeCodeCamp Prediction Function
def predict_message(pred_text):
    text_sequence = tokenizer.texts_to_sequences([pred_text])
    text_padded = pad_sequences(text_sequence, maxlen=max_len, padding='post')
    
    probability = model.predict(text_padded)[0][0]
    
    if probability >= 0.5:
        label = "spam"
    else:
        label = "ham"
        
    return [float(probability), label]
