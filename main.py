import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from fastapi import FastAPI, UploadFile, Form, HTTPException
from pymongo import MongoClient, errors
from pymongo.errors import OperationFailure
import tensorflow as tf
from keras import Layer
from keras.api.models import load_model
from keras.src.saving import custom_object_scope
from dotenv import load_dotenv

from registro_facial import registro_facial, login_captura_facial, consultar_imagen_usuario_2

app = FastAPI()

MONGO_CLIENT = None
facenet = None

def conectar_mongobb(MONGO_HOST = "localhost", MONGO_PORT = "27017", MONGO_DB = "Ejemplo", MONGO_TIMEOUT = 1000):
    try:
        load_dotenv()
        #MONGO_URI = "mongodb://localhost:27017"
        global MONGO_CLIENT
        MONGO_CLIENT = MongoClient(os.getenv("MONGO_URI"))
        try:
            print("Conectado a Mongo", str(MONGO_CLIENT))
            return MONGO_CLIENT
        except OperationFailure as error:
            return "Error en la operación "+ str(error)
    except errors.ServerSelectionTimeoutError as err:
        return "Tiempo excedido ->" + str(err)

def load_facenet_model():
    try:
        with custom_object_scope({'L2Normalization': L2Normalization}):
            global facenet
            archivo = os.path.join(os.path.dirname(__file__), './models/facenet_keras.h5')
            #archivo = os.path.join(os.path.dirname(__file__), 'facenet512.h5')
            facenet = load_model(archivo, custom_objects={'tf': tf})
            # (None, 160, 160, 3) entrada: imagen 160x160 RGB
            # (None, 128) salida: vector 128 elementos
            print("facenet cargado correctamente", str(facenet))
            return facenet
    except EOFError as e:
        print("Error al cargar el modelo:", e)
        return None

class L2Normalization(Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

@app.on_event("startup")
def startup_event():
    conectar_mongobb()
    load_facenet_model()

# Endpoint para registrar un usuario
@app.post("/registro/")
async def registrar_usuario(usuario: str = Form(...), photo: str = Form(...)):
    try:
        if not usuario or not photo:
            raise HTTPException(status_code=400, detail="Faltan datos obligatorios")

        img_id = registro_facial(usuario, photo, MONGO_CLIENT)

        if img_id is None:
            raise HTTPException(status_code=422, detail="No se detectaron rostros")
        elif img_id is False:
            raise HTTPException(status_code=422, detail="Error al decodificar la imagen")

        return {"success": True, "img_id": str(img_id)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")


# Endpoint para iniciar sesión con reconocimiento facial
@app.post("/login/")
async def login_usuario(usuario: str = Form(...), photo: str = Form(...)):
    try:
        # Obtener la imagen registrada desde la base de datos
        user_face = consultar_imagen_usuario_2(imagen_id=usuario, con=MONGO_CLIENT)
        if user_face is False or user_face.size == 0:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        if user_face is None:
            raise HTTPException(status_code=500, detail="Error al recuperar la imagen")
        # Pasar directamente los datos recibidos a `registro_facial`
        result = login_captura_facial(user_face, photo, facenet)

        # Validar el resultado de `registro_facial`
        if result is True:
            return {"success": True, "result": "A"}
        elif result is False:
            return {"success": False, "result": "N"}
        elif result == -100:
            raise HTTPException(status_code=422, detail="Error al decodificar la imagen")
        elif result == -200:
            raise HTTPException(status_code=422, detail="No se detectaron rostros")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")

@app.head("/check")
@app.get("/check")
async def health_check():
    return {"status": "OK"}