import csv

# Ruta al archivo CSV
archivo_csv = '..\Datasets\hateval2019\hateval2019_es_test.csv'

# Abrir el archivo CSV en modo de lectura con el conjunto de caracteres especificado
with open(archivo_csv, 'r', encoding='utf-8') as archivo:
    # Crear un objeto lector de CSV
    lector_csv = csv.reader(archivo)

    # Iterar sobre las filas e imprimir los datos
    for fila in lector_csv:
        print(fila)
