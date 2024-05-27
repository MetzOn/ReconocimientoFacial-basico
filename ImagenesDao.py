from ConexionBD import Conexion
from imagenDTO import Imagen
import cv2
import numpy as np

class ImagenesDao:
    def __init__(self) :
        self.conexion_manager=Conexion()


    def insertarDatosImagen(self,imag):
            con=None
            cursor=None
            if isinstance(imag, Imagen):
                
                nombre=imag.get_nombreI()
                contenido=imag.get_contenidoI()
                Id_Sospechoso= imag.get_idSIma()

            try:
                con=self.conexion_manager.conectar()
                con.autocommit=False
                cursor=con.cursor()
                sql="INSERT INTO imagenes (nombre, contenido, persona_id) VALUES (%s, %s, %s)"
                cursor.execute(sql, (nombre, contenido, Id_Sospechoso))
                con.commit()
            except Exception as e:
                print(f'Error al agregar Imagen: {e}')
            finally:
                self.conexion_manager.desconectar(con, cursor)
                
    def ObtenerContenidoNombresSosp(self):
        con=None
        cursor=None
        facesData = []
        labels = []
        sospechoso_name_mapping = {}  # Un diccionario para hacer un seguimiento de los nombres de los sospechosos
        current_label = 0
        try:
            con = self.conexion_manager.conectar()
            cursor = con.cursor()
            sql = ''' SELECT imagenes.contenido, persona.nombre
                      FROM imagenes
                      INNER JOIN persona ON imagenes.persona_id = persona.id'''
            cursor.execute(sql)
            for contenido_i, nombre_s in cursor:
                # Procesar la imagen desde el contenido binario
                image_array = np.frombuffer(contenido_i, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises
                # Agregar la imagen
                facesData.append(image)
                # Asociar una etiqueta única al nombre del sospechoso
                if nombre_s not in sospechoso_name_mapping:
                    sospechoso_name_mapping[nombre_s] = current_label
                    current_label += 1
                labels.append(sospechoso_name_mapping[nombre_s])
            
            return facesData, labels, sospechoso_name_mapping
        
        except Exception as e:
            print(f'Error al obtener Imagenes: {e}')
            return None
        finally:
            self.conexion_manager.desconectar(con, cursor)
            