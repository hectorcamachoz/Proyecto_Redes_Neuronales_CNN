
import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model("model_CNN2.h5")

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Clases (ajusta según tu modelo si no son dígitos)
class_names = [str(i) for i in range(10)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Imagen original para visualización
    display_frame = frame.copy()

    # Preprocesamiento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (120, 120))  # según cómo entrenaste el modelo
    input_img = resized / 255.0  # Normalizar como en entrenamiento
    input_img = input_img.reshape(1, 120, 120, 1)  # añadir batch y canal

    # Clasificación
    predictions = model.predict(input_img, verbose=0)
    predicted_class = np.argmax(predictions)
    probabilities = predictions[0]

    # Mostrar la imagen que entra al modelo (en ventana aparte)
    cv2.imshow("Input to CNN (120x120)", resized)

    # Escribir probabilidades sobre la imagen original
    y0 = 30
    dy = 25
    for i, prob in enumerate(probabilities):
        color = (0, 255, 0) if i == predicted_class else (0, 0, 255)  # verde si es la clase predicha
        text = f"{class_names[i]}: {prob:.2f}"
        cv2.putText(display_frame, text, (10, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Mostrar la imagen original con texto
    cv2.imshow("Webcam Classification", display_frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
