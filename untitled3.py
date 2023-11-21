import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import nltk
from gensim.models import KeyedVectors

# Cargar los conjuntos de datos
train_df = pd.read_csv('../Datasets/hateval2019/hateval2019_es_train.csv')
dev_df = pd.read_csv('../Datasets/hateval2019/hateval2019_es_dev.csv')
test_df = pd.read_csv('../Datasets/hateval2019/hateval2019_es_test.csv')


# Preprocesamiento del texto
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Aplicar preprocesamiento a los conjuntos de datos
train_df['preprocessed_text'] = train_df['text'].apply(preprocess_text)
dev_df['preprocessed_text'] = dev_df['text'].apply(preprocess_text)
test_df['preprocessed_text'] = test_df['text'].apply(preprocess_text)

# Cargar el modelo de embeddings preentrenados (por ejemplo, Word2Vec)
filename = "../Datasets/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin"
word2vec_model = KeyedVectors.load_word2vec_format(filename, binary=True)

# Función para obtener el embedding promedio de un tweet
def get_average_embedding(text, model):
    words = text.split()
    embeddings = [model.get_vector(word) for word in words if word in model.index_to_key]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return None

# Aplicar la función a los conjuntos de datos
train_df['embedding'] = train_df['preprocessed_text'].apply(lambda x: get_average_embedding(x, word2vec_model))
dev_df['embedding'] = dev_df['preprocessed_text'].apply(lambda x: get_average_embedding(x, word2vec_model))
test_df['embedding'] = test_df['preprocessed_text'].apply(lambda x: get_average_embedding(x, word2vec_model))

# Dividir el conjunto de entrenamiento en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(train_df['embedding'].dropna().to_list(), train_df['HS'].dropna(), test_size=0.2, random_state=42)

# Dimension del Embedding
embedding_dim = 300

# Construir el modelo de red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(embedding_dim,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, validation_data=(np.array(X_val), np.array(y_val)))

# Evaluar el modelo en el conjunto de prueba
X_test, y_test = test_df['embedding'].dropna().to_list(), test_df['HS'].dropna()
predictions = (model.predict(np.array(X_test)) > 0.5).astype(int)
accuracy = accuracy_score(y_test, predictions)
print(f'Precisión del modelo en el conjunto de prueba: {accuracy}')


print("Train Loss:", history.history['loss'])
print("Validation Loss:", history.history['val_loss'])
print("Train Accuracy:", history.history['accuracy'])
print("Validation Accuracy:", history.history['val_accuracy'])

# Graficar la pérdida y la precisión en función de las épocas
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Pérdida en función de las épocas')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Precisión en función de las épocas')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.show()
