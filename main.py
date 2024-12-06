from fastapi import FastAPI, UploadFile, Form, HTTPException
from registro_facial import registro_facial, login_captura_facial, consultar_imagen_usuario_2

app = FastAPI()


# Endpoint para registrar un usuario
@app.post("/registro/")
async def registrar_usuario(usuario: str = Form(...), photo: str = Form(...)):
    try:

        if not usuario or not photo:
            raise HTTPException(status_code=400, detail="Faltan datos obligatorios")


        img_id = registro_facial(usuario, photo)

        if img_id is None:
            raise HTTPException(status_code=422, detail="No se detectaron rostros")
        elif img_id is False:
            raise HTTPException(status_code=422, detail="Error al decodificar la imagen")

        return {"success": True, "img_id": str(img_id)}

    except HTTPException as e:
        return {"success": False, "error": e.detail}
    except Exception as e:

        return {"success": False, "error": "Error inesperado: " + str(e)}


# Endpoint para iniciar sesión con reconocimiento facial
@app.post("/login/")
async def login_usuario(usuario: str = Form(...), photo: str = Form(...)):
    # Obtener la imagen registrada desde la base de datos
    user_face = consultar_imagen_usuario_2(imagen_id=usuario)
    if user_face is None or user_face.size == 0:
        return {"success": False, "error": "Usuario no encontrado"}

    try:
        # Pasar directamente los datos recibidos a `registro_facial`
        result = login_captura_facial(user_face, photo)

        # Validar el resultado de `registro_facial`
        if result is True:
            return {"success": True, "result": "A"}
        elif result is False:
            return {"success": False, "error": "N"}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.head("/check")
@app.get("/check")
async def health_check():
    return {"status": "OK"}