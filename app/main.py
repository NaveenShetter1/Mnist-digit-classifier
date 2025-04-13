from fastapi import FastAPI, UploadFile, File

from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io


app=FastAPI()
model=load_model("mnist_model.h5")


@app.post("/predict/")
async def predict(file: UploadFile=File(...)):
    image=Image.open(io.BytesIO(await file.read())).convert("L").resize((28,28))
    image_array=np.array(image).reshape(1,784)/255.0    
    prediction=model.predict(image_array)
    predicted_class=int(np.argmax(prediction))
    return {"prediction": predicted_class}
