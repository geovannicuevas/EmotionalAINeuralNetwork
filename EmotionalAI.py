import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data from JSONL file
def load_data_from_jsonl(file_path):
    text_data = []
    category_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            text_data.append(data['text'])
            category_data.append(category_mapping.get(data['label']))
    return text_data, category_data

# Define category mappings
category_mapping = {
    '1': 'Bias or Abuse',
    '2': 'Career',
    '3': 'Medication',
    '4': 'Relationship',
    '5': 'Alienation'
}

# Merge JSONL files
def merge_jsonl_files(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        merged_data = [json.loads(line) for line in f1] + [json.loads(line) for line in f2]

    with open(output_file, 'w') as output_file:
        for line in merged_data:
            json.dump(line, output_file)
            output_file.write('\n')

# Paths to JSONL files
jsonl_file_path = '/Users/geocuevas/Downloads/mental_health.jsonl'
jsonl_file_path2 = '/Users/geocuevas/Downloads/mental_health-2.jsonl'
merged_file = '/Users/geocuevas/PycharmProjects/ResearchData/.venv/lib/merged_file.jsonl'

# Merge JSONL files
merge_jsonl_files(jsonl_file_path, jsonl_file_path2, merged_file)

# Load data and encode labels
x, y = load_data_from_jsonl(merged_file)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = keras.Sequential()
model.add(layers.LSTM(64, input_shape=(None, 28)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10))
print(model.summary())
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_validate, y_validate = x_test[:-10], y_test[:-10]
x_test, y_test = x_test[-10:], y_test[-10:]
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)
for i in range(10):
    result = tf.argmax(model.predict(tf.expand_dims(x_test[i], 0)), axis=1)
    print(result.numpy(), y_test[i])
# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
x_sequences = tokenizer.texts_to_sequences(x)
max_length = 100
x_padded = pad_sequences(x_sequences, maxlen=max_length)

# Split data into train, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(x_padded, y_encoded, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Initialize models for each category
models = {}
for category_name in category_mapping.values():
    num_classes = len(category_mapping)
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    models[category_name] = model

# Train and evaluate models for each category
for category_name, model in models.items():
    print(f"Training model for {category_name}...")
    model.fit(x_train, y_train, epochs=100, batch_size=500, validation_data=(x_val, y_val), verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy for {category_name}: {accuracy}")

