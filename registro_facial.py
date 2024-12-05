import base64
import os

from dotenv import load_dotenv

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import gridfs
import tensorflow as tf
import cv2
from keras import Layer
from keras.api.models import load_model
from keras.src.saving import custom_object_scope
from pymongo import MongoClient, errors
from pymongo.errors import OperationFailure
from bson import ObjectId

def load_mobilenet():
    # Define el archivo
    archivo = os.path.join(os.path.dirname(__file__), './models/graph.pb')
    # Verifica si el archivo existe
    if not os.path.exists(archivo):
        print(f"Error: El archivo '{archivo}' no se encuentra en el directorio actual.")
        exit(1)

    # Intenta cargar el archivo
    try:
        with tf.io.gfile.GFile(archivo, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

            with tf.Graph().as_default() as mobilenet:
                tf.import_graph_def(graph_def, name='')
                return mobilenet

    except Exception as e:
        print(f"Error al cargar el archivo '{archivo}': {e}")
        exit(1)

load_dotenv()

def conectar_mongobb(MONGO_HOST = "localhost", MONGO_PORT = "27017", MONGO_DB = "Ejemplo", MONGO_TIMEOUT = 1000):
    try:
        #MONGO_URI = "mongodb://localhost:27017"
        MONGO_CLIENT = MongoClient(os.getenv("MONGO_URI"))
        try:
            return MONGO_CLIENT
        except OperationFailure as error:
            return "Error en la operación "+ str(error)
    except errors.ServerSelectionTimeoutError as err:
        return "Tiempo excedido ->" + str(err)

def load_facenet_model():
    try:
        with custom_object_scope({'L2Normalization': L2Normalization}):
            archivo = os.path.join(os.path.dirname(__file__), './models/facenet_keras.h5')
            #archivo = os.path.join(os.path.dirname(__file__), 'facenet512.h5')
            facenet = load_model(archivo, custom_objects={'tf': tf})
            # (None, 160, 160, 3) entrada: imagen 160x160 RGB
            # (None, 128) salida: vector 128 elementos
            return facenet
    except EOFError as e:
        print("Error al cargar el modelo:", e)
        return None

"""
def load_vgg_model():
    try:
        with custom_object_scope({'L2Normalization': L2Normalization}):
            archivo = os.path.join(os.path.dirname(__file__), './models/vgg.h5')
            vgg = load_model(archivo, custom_objects={'tf': tf})
            print(vgg.input_shape)
            return vgg
    except EOFError as e:
        print("Error al cargar el modelo:", e)
        return None
"""

def detect_faces(image, score_threshold=0.7):
    global boxes, scores
    mobilenet = load_mobilenet()
    (imh, imw) = image.shape[:-1]
    img = np.expand_dims(image, axis=0)
    # Inicializar mobilenet
    sess = tf.compat.v1.Session(graph=mobilenet)
    image_tensor = mobilenet.get_tensor_by_name('image_tensor:0')
    boxes = mobilenet.get_tensor_by_name('detection_boxes:0')
    scores = mobilenet.get_tensor_by_name('detection_scores:0')
    # Predicción (detección)
    (boxes, scores) = sess.run([boxes, scores], feed_dict={image_tensor: img})
    # Reajustar tamaños boxes, scores
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)
    # Depurar bounding boxes
    idx = np.where(scores >= score_threshold)[0]
    # Crear bounding boxes
    bboxes = []
    for index in idx:
        ymin, xmin, ymax, xmax = boxes[index, :]
        (left, right, top, bottom) = (xmin * imw, xmax * imw, ymin * imh, ymax * imh)
        left, right, top, bottom = int(left), int(right), int(top), int(bottom)

        bboxes.append([left, right, top, bottom])
    return bboxes

# Funcion que dibuja sobre la imagen el resultado de la deteccion de rostros
def draw_box(image, box, color, line_width=6):
    if box == []:
        return image
    else:
        cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), color, line_width)
    return image

"""
Función que extrae la porción de la imagen que contiene el rostro (detectada por detect_faces) 
y luego redimensionarla a una imagen de 160×160 de acuerdo a los requerimientos de FaceNet.
"""
def extract_faces(image,bboxes,new_size=(160,160)):
    cropped_faces = []
    for box in bboxes:
        left, right, top, bottom = box
        face = image[top:bottom,left:right]
        cropped_faces.append(cv2.resize(face,dsize=new_size))
    return cropped_faces

class L2Normalization(Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

# Funcion que guarda la imagen del registro en mongo
def imagen_register_mongodb(bd, coleccion, rostro, name):
    #rostro = load_image("rostros_planos/", "c.jpg")
    #plt.imshow(cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB))
    _, buffer = cv2.imencode('.jpg', rostro)  # Convierte la imagen a binario JPEG
    img_data = buffer.tobytes()  # Convierte el buffer a bytes
    con = conectar_mongobb()
    try:
        db = con[bd]
        fs = gridfs.GridFS(db, collection=coleccion)
        img_id = fs.put(img_data, filename=name)
        print(f"Imagen almacenada con ID: {img_id}")
        return img_id
        con.close()
    except Exception as e:
        print(f"Ocurrió un error al almacenar la imagen: {e}")
        con.close()
        return None

# Funcion que registra el rostro, lo captura y almacena de acuerdo a los requerimientos de Facenet
def registro_facial(usuario, frame):

    # Decodificar la imagen base64
    photo_data = frame.split(",")[1]  # Elimina el prefijo de tipo ###########
    img_data = base64.b64decode(photo_data)

    # Convertir la imagen decodificada a un formato adecuado para OpenCV
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    # Verificar si la imagen se decodificó correctamente
    if frame is None:
        return False

    # Asegúrate de que la imagen tenga 3 canales (RGB)
    if frame.shape[2] == 4:  # Si tiene 4 canales (BGRA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    # Verificar si se detectaron rostros
    bboxes = detect_faces(frame)
    if not bboxes:
        return None

    name = usuario + ".jpg"

    for box in bboxes:
        frame = draw_box(frame, box, (0, 2575, 0))  # Dibuja un cuadro alrededor de cada rostro detectado

    # Extraer las caras del marco
    #faces = extract_faces(frame, bboxes)
    faces = extract_faces(frame, bboxes)
    #rostro = faces[0] if faces else frame  # Usa la primera cara o el marco completo si no hay caras
    rostro = faces[0]
    cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)

    # Registrar la imagen en MongoDB
    img_id = imagen_register_mongodb("FACEGUARD", "Register", rostro, name)

    return img_id

def calcular_embedding(model, face):
    # Normalizar la imagen restando a cada pixel el valor promedio y dividiéndolo
    # entre la desviación estándar (face = (face-mean) / std).
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)

    # Calculo del embedding con una predicción de Facenet
    embedding = model.predict(face)
    return embedding

# Funcion que obtiene desde Mongo la imagen del ususario que se quiere loggear
def consultar_imagen_usuario(bd="FACEGUARD", coleccion="Register", imagen_id=""):
    # Conectar a MongoDB
    con = conectar_mongobb()
    try:
        db = con[bd]
        fs = gridfs.GridFS(db, collection=coleccion)
        # Buscar el archivo por nombre de usuario
        imagen_doc = db[coleccion+".files"].find_one({"_id": ObjectId(imagen_id)})  # Busca en la colección de archivos
        if imagen_doc:
            # Obtener el ID del archivo
            img_id = imagen_doc['_id']
            # Recuperar la imagen
            img_data = fs.get(img_id).read()
            #print(f"Imagen recuperada para el usuario: {nombre_usuario}")

            # Decodificar los bytes a una imagen de OpenCV
            user_face = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)

            #plt.imshow(user_face)
            #plt.show()

            return user_face  # Retorna los datos de la imagen en formato OpenCV
        else:
            con.close()
            print("El usuario no existe")
            return False
    except Exception as e:
        print(f"Ocurrió un error al recuperar la imagen: {e}")
        con.close()
        return None

def consultar_imagen_usuario_2(bd="FACEGUARD", coleccion="Register", imagen_id=""):
    # Conectar a MongoDB
    con = conectar_mongobb()
    try:
        db = con[bd]
        fs = gridfs.GridFS(db, collection=coleccion)
        # Buscar el archivo por nombre de usuario
        imagen_doc = db[coleccion+".files"].find_one({"filename": imagen_id+".jpg"})  # Busca en la colección de archivos
        if imagen_doc:
            # Obtener el ID del archivo
            img_id = imagen_doc['_id']
            print(img_id)
            # Recuperar la imagen
            img_data = fs.get(img_id).read()
            #print(f"Imagen recuperada para el usuario: {nombre_usuario}")

            # Decodificar los bytes a una imagen de OpenCV
            user_face = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)

            #plt.imshow(user_face)
            #plt.show()

            return user_face  # Retorna los datos de la imagen en formato OpenCV
        else:
            con.close()
            print("El usuario no existe")
            return False
    except Exception as e:
        print(f"Ocurrió un error al recuperar la imagen: {e}")
        con.close()
        return None

def login_captura_facial(user_face, frame):
    #num = 0
    # Decodificar la imagen base64
    photo_data = frame.split(",")[1]  # Elimina el prefijo de tipo
    img_data = base64.b64decode(photo_data)

    # Convertir la imagen decodificada a un formato adecuado para OpenCV
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    # Verificar si la imagen se decodificó correctamente
    if frame is None:
        return -100

    # Asegúrate de que la imagen tenga 3 canales (RGB)
    if frame.shape[2] == 4:  # Si tiene 4 canales (BGRA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    # Verificar si se detectaron rostros
    bboxes = detect_faces(frame)
    if not bboxes:
        return -200

    for box in bboxes:
        frame = draw_box(frame, box, (0, 2575, 0))

    # Extraer las caras del marco
    #faces = extract_faces(frame, bboxes)
    faces = extract_faces(frame, bboxes)
    # rostro = faces[0] if faces else frame  # Usa la primera cara o el marco completo si no hay caras
    login_face = faces[0]
    cv2.cvtColor(login_face, cv2.COLOR_BGR2RGB)

    #plt.imshow(login_face)
    #plt.show()

    facenet = load_facenet_model()
    user_embeddingf = calcular_embedding(facenet, user_face)
    login_embeddingf = calcular_embedding(facenet, login_face)

    #vgg = load_vgg_model()
    #user_embeddingv = calcular_embedding(vgg, user_face)
    #login_embeddingv = calcular_embedding(vgg, login_face)

    # Comparar los embeddings de los dos rostros
    distf = np.linalg.norm(user_embeddingf - login_embeddingf)
    print("facenet: " + str(distf))
    umbral_dist = 0.250000000
    if distf < umbral_dist:
        print("Acceso concedido: Los rostros coinciden.")
        return True
        #num=num+1
    else:
        print("Acceso denegado: Los rostros no coinciden.")
        return False

    """
    distv = np.linalg.norm(user_embeddingv - login_embeddingv)
    print("vgg: " + str(distv))
    umbral_dist = 0.250000000
    if distv < umbral_dist:
        print("Acceso concedido: Los rostros coinciden.")
        #return True
        num = num + 1
    else:
        print("Acceso denegado: Los rostros no coinciden.")
        #return False
    if num > 1:
        return True
    else:
        return False
    """