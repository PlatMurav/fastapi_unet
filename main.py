from model import Segmentator
from fastapi import FastAPI, File, UploadFile, Response
import os
import tensorflow as tf
import numpy as np
import cv2
import errno

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()

    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    image = segmentator.preprocess_image(img)

    prd_mask = model.predict(image)
    final_mask = segmentator.postprocess_image(prd_mask)

    ret, buffer = cv2.imencode('.png', final_mask.numpy())

    # Convert the buffer to a byte string
    byte_string = buffer.tobytes()

    return Response(content=byte_string, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    segmentator = Segmentator()

    filename = 'model_t800.keras'
    if not os.path.exists(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    else:
        model = tf.keras.models.load_model('model_t800.keras')

    uvicorn.run(app, host='127.0.0.1', port=8000)


