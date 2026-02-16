from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 300

app = FastAPI()

model = tf.keras.models.load_model("models/brain_tumor_model.h5")

class_names = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Correct preprocessing (same as training)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4)
    }
