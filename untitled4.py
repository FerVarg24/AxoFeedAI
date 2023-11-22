import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
import matplotlib.pyplot as plt

# Cargar datos
train_data = pd.read_csv('../Datasets/hateval2019/hateval2019_es_train.csv')
dev_data = pd.read_csv('../Datasets/hateval2019/hateval2019_es_dev.csv')
test_data = pd.read_csv('../Datasets/hateval2019/hateval2019_es_test.csv')

# Preprocesar texto
max_words = 10000  # Número máximo de palabras a considerar
max_len = 100  # Longitud máxima de la secuencia de palabras

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data['text'])

X_train = tokenizer.texts_to_sequences(train_data['text'])
X_dev = tokenizer.texts_to_sequences(dev_data['text'])
X_test = tokenizer.texts_to_sequences(test_data['text'])

X_train = pad_sequences(X_train, maxlen=max_len)
X_dev = pad_sequences(X_dev, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Codificar etiquetas
label_encoder = LabelEncoder()
y_train_HS = label_encoder.fit_transform(train_data['HS'])
y_train_TR = label_encoder.fit_transform(train_data['TR'])
y_train_AG = label_encoder.fit_transform(train_data['AG'])

# Codificar etiquetas para datos de desarrollo
y_dev_HS = label_encoder.transform(dev_data['HS'])
y_dev_TR = label_encoder.transform(dev_data['TR'])
y_dev_AG = label_encoder.transform(dev_data['AG'])


# Construir modelo
embedding_dim = 300  # Dimensión de los embeddings
vocab_size = min(max_words, len(tokenizer.word_index) + 1)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))  # Para la tarea de clasificación binaria

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train_HS, epochs=5, batch_size=32, validation_data=(X_dev, y_dev_HS))

# Evaluar en datos de prueba
y_pred = model.predict(X_test)


# Visualizar la pérdida y la precisión durante el entrenamiento
plt.figure(figsize=(12, 4))

# Plotear la pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotear la precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Imprimir resultados de prueba
print("Results on test data:")
print(y_pred)

# Loop para clasificar palabras ingresadas
while True:
    input_text = input("Ingrese una palabra o frase (o escriba 'salir' para terminar): ")
    
    if input_text.lower() == 'salir':
        break
    
    # Preprocesar la entrada
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(input_sequence, maxlen=max_len)
    
    # Realizar la predicción
    prediction = model.predict(input_sequence)
    
    # Clasificar la predicción
    if prediction[0, 0] >= 0.5:
        print("Mensaje de odio")
    else:
        print("Mensaje inofensivo")

# Puedes ajustar este código para incorporar las otras etiquetas (TR, AG) y realizar ajustes según sea necesario.
