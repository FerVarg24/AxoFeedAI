let model;
let wordIndex;

window.onload = async function() {
    model = await tf.loadLayersModel('model.json'); // Asegúrate de cambiar esto por la ruta correcta a tu modelo
    await loadWordIndex(); // Cargar el diccionario word_index
    console.log("Modelo y word_index cargados");
};

// Función para cargar el diccionario word_index
async function loadWordIndex() {
    const response = await fetch('metadata.json'); // Cambia esto por la ruta correcta a tu archivo JSON
    const data = await response.json();
    wordIndex = data.word_index;
}

// Función para convertir texto a secuencia
function textToSequence(text) {
    const words = text.toLowerCase().split(' ');
    const sequence = words.map(word => wordIndex[word] || 0);
    return sequence;
}

// Función para rellenar secuencias
function padSequence(sequence, maxLength) {
    while (sequence.length < maxLength) {
        sequence.unshift(0); // Agrega ceros al principio
    }
    return sequence.slice(-maxLength); // Asegúrate de que la secuencia no sea más larga que maxLength
}

async function evaluateText() {
    let inputText = document.getElementById('inputText').value;

    let processedInput = processInput(inputText);
    processedInput = tf.tensor2d([processedInput]); // Convierte la secuencia a un tensor 2D

    // Realizar predicción
    let prediction = await model.predict(processedInput).data();

    // Interpretar la predicción y mostrar resultados
    let resultText = prediction[0] >= 0.5 ? "Mensaje de odio" : "Mensaje inofensivo";
    document.getElementById('result').innerText = resultText;
}

// Procesa el texto de entrada
function processInput(inputText) {
    let sequence = textToSequence(inputText);
    sequence = padSequence(sequence, 100); // Asegúrate de que la longitud sea igual a max_len
    return sequence;
}
