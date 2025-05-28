import numpy as np

# Dimensiones de la red
entrada_dim = 25 
oculta_dim = 5
salida_dim = 3

# Inicialización de pesos y sesgos
pesos_entrada_oculta = np.random.normal(0, 0.1, (entrada_dim, oculta_dim))
pesos_oculta_salida = np.random.normal(0, 0.1, (oculta_dim, salida_dim))
sesgo_oculta = np.zeros(oculta_dim)
sesgo_salida = np.zeros(salida_dim)

# Funciones de activación
def activacion_sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return x * (1 - x)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

# Normalización de entradas
def normalizar(entrada):
    min_val = np.min(entrada)
    max_val = np.max(entrada)
    if max_val - min_val == 0:
        return entrada
    return (entrada - min_val) / (max_val - min_val)

# Paso hacia adelante
def paso_hacia_adelante(entrada):
    entrada_norm = normalizar(entrada)
    activacion_oculta = activacion_sigmoide(np.dot(entrada_norm, pesos_entrada_oculta) + sesgo_oculta)
    salida = softmax(np.dot(activacion_oculta, pesos_oculta_salida) + sesgo_salida)
    return salida

# Generación de patrones como matrices 5x5
def generar_circulo():
    patron = np.zeros((5, 5))
    centro = (2, 2)
    for i in range(5):
        for j in range(5):
            distancia = np.sqrt((i - centro[0])**2 + (j - centro[1])**2)
            if 0.8 <= distancia <= 1.8:
                patron[i, j] = 1
    ruido = np.random.normal(0, 0.5, (5, 5))  
    patron += ruido
    return patron.flatten()

def generar_cuadrado():
    patron = np.zeros((5, 5))
    for i in [0, 4]:
        patron[i, 1:4] = 1
    for j in [0, 4]:
        patron[1:4, j] = 1
    ruido = np.random.normal(0, 0.5, (5, 5))
    patron += ruido
    return patron.flatten()

def generar_triangulo():
    patron = np.zeros((5, 5))
    patron[4, 1:4] = 1
    patron[3, 2] = 1
    patron[2, 2] = 1
    ruido = np.random.normal(0, 0.5, (5, 5))
    patron += ruido
    return patron.flatten()

# Entrenamiento
def entrenar(entradas, etiquetas, tasa_aprendizaje=0.1, epocas=10): 
    global pesos_entrada_oculta, pesos_oculta_salida, sesgo_oculta, sesgo_salida
    for _ in range(epocas):
        for entrada, etiqueta in zip(entradas, etiquetas):
            entrada_norm = normalizar(entrada)
            activacion_oculta = activacion_sigmoide(np.dot(entrada_norm, pesos_entrada_oculta) + sesgo_oculta)
            salida = softmax(np.dot(activacion_oculta, pesos_oculta_salida) + sesgo_salida)

            etiqueta_one_hot = np.zeros(salida_dim)
            etiqueta_one_hot[etiqueta] = 1

            error_salida = salida - etiqueta_one_hot
            error_oculta = np.dot(error_salida, pesos_oculta_salida.T) * derivada_sigmoide(activacion_oculta)

            pesos_oculta_salida -= tasa_aprendizaje * np.outer(activacion_oculta, error_salida)
            pesos_entrada_oculta -= tasa_aprendizaje * np.outer(entrada_norm, error_oculta)
            sesgo_salida -= tasa_aprendizaje * error_salida
            sesgo_oculta -= tasa_aprendizaje * error_oculta

# Prueba y evaluación
def probar_y_adivinar(num_pruebas=20): 
    aciertos = 0
    for _ in range(num_pruebas):
        categoria = np.random.choice([0, 1, 2])
        if categoria == 0:
            patron = generar_circulo()
            etiqueta_real = 0
        elif categoria == 1:
            patron = generar_cuadrado()
            etiqueta_real = 1
        else:
            patron = generar_triangulo()
            etiqueta_real = 2

        salida = paso_hacia_adelante(patron)
        prediccion = np.argmax(salida)

        print("\nPatrón generado (matriz 5x5):")
        print(np.round(patron.reshape(5,5), 2))
        print(f"Predicción: {'Círculo' if prediccion == 0 else 'Cuadrado' if prediccion == 1 else 'Triángulo'}")
        print(f"Etiqueta real: {'Círculo' if etiqueta_real == 0 else 'Cuadrado' if etiqueta_real == 1 else 'Triángulo'}")

        if prediccion == etiqueta_real:
            aciertos += 1

    porcentaje_acertacion = (aciertos / num_pruebas) * 100
    print(f"\nPorcentaje de aciertos: {porcentaje_acertacion:.2f}%")
    return porcentaje_acertacion

# Generar datos de entrenamiento
num_ejemplos = 70
datos_circulo = [generar_circulo() for _ in range(num_ejemplos)]
datos_cuadrado = [generar_cuadrado() for _ in range(num_ejemplos)]
datos_triangulo = [generar_triangulo() for _ in range(num_ejemplos)]
datos_entrenamiento = datos_circulo + datos_cuadrado + datos_triangulo
etiquetas_entrenamiento = [0] * num_ejemplos + [1] * num_ejemplos + [2] * num_ejemplos

# Entrenar y probar
entrenar(datos_entrenamiento, etiquetas_entrenamiento)
probar_y_adivinar(num_pruebas=10)
