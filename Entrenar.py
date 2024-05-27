import cv2
import numpy as np
from ImagenesDao import ImagenesDao  # Importa la clase ImagenesDao para acceder a la base de datos

class Entrenamiento:
    def __init__(self):
        self.dao_imagenes = ImagenesDao()  # Crea una instancia de ImagenesDao para acceder a los métodos de la base de datos
        self.Diccionario={}
        self.etiqueta_sospechoso_mapping = {etiqueta: nombre for nombre, etiqueta in self.Diccionario.items()}
        self.ListImagenesC=[]
        self.ListImagenesN=[]
        self.NombresSos=[]
        self.indicesReconocimiento=[]

    def entrenarSistema(self):
            facesData,self.indicesReconocimiento,self.Diccionario=self.dao_imagenes.ObtenerContenidoNombresSosp()
            self.etiqueta_sospechoso_mapping = {etiqueta: nombre for nombre, etiqueta in self.Diccionario.items()}
            print("Diccionario de entrenamiento:", self.etiqueta_sospechoso_mapping)

            model = cv2.face.LBPHFaceRecognizer_create()
            print("Entrenando...")
            model.train(facesData, np.array(self.indicesReconocimiento))

            # Guardar el modelo entrenado en un archivo XML
            model.write("ModeloFacesFrontalData2023.xml")
            
            print("Entrenamiento Completado", "El modelo ha sido entrenado y guardado.")
            return self.etiqueta_sospechoso_mapping
    
entrenamiento = Entrenamiento() 
entrenamiento.entrenarSistema()


