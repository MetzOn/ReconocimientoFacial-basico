import cv2
from Entrenar import Entrenamiento
class ReconocimientoFacial:
    def __init__(self):
        # Inicialización de la clase
        # Se reciben los objetos necesarios para el reconocimiento facial
        self.faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face_recognizer= cv2.face.LBPHFaceRecognizer_create()
        self.entrenamiento=Entrenamiento()
        self.etiqueta_sospechoso_mapping = {}  # Mapeo de etiquetas de sospechosos

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Inicializa la captura de video desde la cámara
        self.captura_pausada = False  # Indicador para pausar la captura de video

    def visualizar(self):
        self.face_recognizer.read('ModeloFacesFrontalData2023.xml')
        # Método para visualizar el reconocimiento facial en tiempo real
        while True:  # Bucle infinito para mantener la visualización continua
            ret, frame = self.cap.read()  # Captura un frame de video

            if ret:  # Si se captura correctamente un frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte el frame a escala de grises
                auxFrame = gray.copy()  # Copia del frame en escala de grises
                faces = self.faceClassif.detectMultiScale(gray, 1.3, 5)  # Detecta las caras en el frame

                for (x, y, w, h) in faces:  # Itera sobre las coordenadas de las caras detectadas
                    rostro = auxFrame[y:y + h, x:x + w]  # Recorta la región de la cara
                    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)  # Redimensiona la cara
                    result = self.face_recognizer.predict(rostro)  # Realiza el reconocimiento facial en la cara

                    if result[1] < 80:  # Si la confianza en el reconocimiento es alta
                        nombre = self.etiqueta_sospechoso_mapping.get(result[0], "No encontrado")  # Obtiene el nombre correspondiente a la etiqueta
                        if nombre != "No encontrado":
                            name = nombre
                    else:  # Si la confianza en el reconocimiento es baja
                        name = "Desconocido"

                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Muestra el nombre en el frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibuja un rectángulo alrededor de la cara reconocida

                cv2.imshow('Reconocimiento Facial', frame)  # Muestra el frame con las caras detectadas
                key = cv2.waitKey(1)  # Espera 1 milisegundo por la pulsación de una tecla
                if key == 27: 
                    self.finalizar_video() # Si se presiona la tecla 'Esc' (27 en ASCII), finaliza la visualización
                    break

    # Finaliza la visualización del video

    def finalizar_video(self):
        # Método para finalizar la visualización del video y liberar los recursos de la cámara
        self.cap.release()  # Libera los recursos de la cámara
        cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV


    def finalizar_video(self):
        # Método para finalizar la visualización del video y liberar los recursos de la cámara
        self.cap.release()  # Libera los recursos de la cámara
        cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV

reconocimiento=ReconocimientoFacial()
etiqueta_sospechoso_mapping=reconocimiento.entrenamiento.entrenarSistema()
reconocimiento.etiqueta_sospechoso_mapping = etiqueta_sospechoso_mapping 
print(reconocimiento.etiqueta_sospechoso_mapping)
reconocimiento.visualizar()
