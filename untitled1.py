import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from gensim.models import KeyedVectors
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight

# Función para extraer texto de cada entrada
def extract_text(entry):
    parts = entry.split('||;')
    if len(parts) > 1:
        text = parts[1].strip()
        return text
    else:
        return ""

# Función para limpiar texto
def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

# Función para convertir a minúsculas
def lowercase_text(text):
    return text.lower()

# Función para tokenizar texto
def tokenize_text(text_data, max_length):
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    vocab_size = len(tokenizer.word_index) + 1
    return padded_sequences, vocab_size, tokenizer

# Especifica la ruta de tus embeddings preprocesados
filename = "GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin"
word2vec_model = KeyedVectors.load_word2vec_format(filename, binary=True)

# Lee el contenido del archivo con tu dataset
file_path = '../Datasets/labeled_corpus_6k.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    dataset = file.readlines()

# Procesa cada línea para extraer el texto y la etiqueta
text_data = []
labels = []

for entry in dataset:
    text = extract_text(entry)
    label = int(entry.split('||;')[-1])
    
    text_data.append(text)
    labels.append(label)

# División del Conjunto de Datos
X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.1, random_state=42)

# Preprocesamiento y Tokenización de Datos
max_length = 50  # Longitud máxima de las secuencias
X_train_cleaned = [lowercase_text(clean_text(extract_text(entry))) for entry in X_train]
X_test_cleaned = [lowercase_text(clean_text(extract_text(entry))) for entry in X_test]

X_train_tokenized, vocab_size, tokenizer = tokenize_text(X_train_cleaned, max_length)  # Utiliza el tokenizer devuelto
X_test_tokenized, _, _ = tokenize_text(X_test_cleaned, max_length)  # No necesitas el tokenizer aquí

# Aplicar sobremuestreo solo en los datos de entrenamiento
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_tokenized, y_train)

# Verificar el nuevo balance de clases
unique, counts = np.unique(y_train_resampled, return_counts=True)
print("Recuento de instancias por clase (entrenamiento después de sobremuestreo):", dict(zip(unique, counts)))

# Conversión a matrices NumPy
X_train_resampled_np = np.array(X_train_resampled)
y_train_resampled_np = np.array(y_train_resampled).reshape(-1, 1)  # Reformatear las etiquetas

X_test_tokenized_np = np.array(X_test_tokenized)
y_test_np = np.array(y_test).reshape(-1, 1)  # Reformatear las etiquetas

# Construcción del Modelo con embeddings preprocesados
embedding_dim = 300  # Dimensión del espacio de embedding

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in word2vec_model:
        embedding_matrix[i] = word2vec_model[word]

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0001), recurrent_regularizer=l2(0.0001))),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
])
custom_optimizer = Adam(learning_rate=0.0001) 
model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento del Modelo con los datos sobremuestreados
history = model.fit(X_train_resampled_np, y_train_resampled_np, epochs=95, batch_size=128, validation_data=(X_test_tokenized_np, y_test_np))

# Historial de entrenamiento
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Graficar la pérdida
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Graficar la precisión
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluación del Modelo
loss, accuracy = model.evaluate(X_test_tokenized_np, y_test_np)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Verificación del equilibrio de clases
plt.hist(labels, bins=len(set(labels)))
plt.show()

# Supongamos que 'y_train' es tu conjunto de etiquetas de entrenamiento
# y 'y_test' es tu conjunto de etiquetas de prueba

# Contar las instancias en cada clase para el conjunto de entrenamiento
unique_train, counts_train = np.unique(y_train, return_counts=True)
class_counts_train = dict(zip(unique_train, counts_train))

# Contar las instancias en cada clase para el conjunto de prueba
unique_test, counts_test = np.unique(y_test, return_counts=True)
class_counts_test = dict(zip(unique_test, counts_test))

# Imprimir el recuento de instancias por clase para el conjunto de entrenamiento
print("Recuento de instancias por clase (entrenamiento):", class_counts_train)

# Imprimir el recuento de instancias por clase para el conjunto de prueba
print("Recuento de instancias por clase (prueba):", class_counts_test)

# Función para preprocesar y tokenizar un solo texto
def preprocess_and_tokenize_text(text, tokenizer, max_length):
    cleaned_text = lowercase_text(clean_text(text))
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    return np.array(padded_sequence)

# Ingresar el nuevo texto que deseas clasificar
new_text = "hola"

# Preprocesar y tokenizar el nuevo texto
tokenized_text = preprocess_and_tokenize_text(new_text, tokenizer, max_length)

# Realizar la predicción
prediction = model.predict(tokenized_text)

# Imprimir la predicción
print("Probabilidad de pertenencia a la clase positiva:", prediction[0][0])

# Convertir la probabilidad en una etiqueta de clase
predicted_class = 1 if prediction[0][0] >= 0.5 else 0
print("Clase predicha:", predicted_class)
