
from ImagenesDao import ImagenesDao
from imagenDTO import Imagen
import cv2
import imutils


class CapturarImagen():
    def __init__(self):
        self.imagenes = []
        self.DaoImg=ImagenesDao()
    
    def capturar_camara(self):
        # Permite activar la camara tiempo real
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # Se toma el
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        count = 0
        while True:
            ret, frame = cap.read()  # Capta los frame de los videos
            if ret == False:
                break
            frame = imutils.resize(frame, width=640)  # Se redimenciona el frame por si es muy grande
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Se cambia el frame a blanco y negro para mejor precision
            auxFrame = frame.copy()  # Copia los frames leidos
            faces = faceClassif.detectMultiScale(gray, 1.3, 5)  # detecta la posicion de los rostros
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibuja un rectángulo alrededor de cada rostro
                rostro = auxFrame[y:y + h, x:x + w]  # Recorta la región del rostro
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)  # Redimensiona el rostro a 150x150 píxeles

                # Crea un objeto Imagen y lo agrega a la lista
                imagen = Imagen()
                imagen.set_nombreI('rostro_{}.jpg'.format(count))
                # Convierte la imagen a formato binario
                _, buffer = cv2.imencode('.jpg', rostro)
                imagen.set_contenidoI(buffer.tobytes())
                imagen.set_idSIma(1)  # Reemplaza 1 con el ID de la persona correspondiente
                self.imagenes.append(imagen)

                count += 1
                print(count)
            cv2.imshow('frame', frame)  # Muestra el frame con los rectángulos dibujados
            k = cv2.waitKey(1)
            if k == 27 or count >= 10:  # Permite presionar esc para salir de la toma de datos, o se finaliza con 300 imagenes capturadas
                break
        cap.release()
        cv2.destroyAllWindows()

    def insertar_imagenes_bd(self):
        for imagen in self.imagenes:
            self.DaoImg.insertarDatosImagen(imagen)


if __name__ == "__main__":
    capturar = CapturarImagen()
    capturar.capturar_camara()
    capturar.insertar_imagenes_bd()