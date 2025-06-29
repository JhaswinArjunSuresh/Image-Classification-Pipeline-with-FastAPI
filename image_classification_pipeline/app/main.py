from fastapi import FastAPI, UploadFile, File, HTTPException
from .classifier import ImageClassifier

app = FastAPI()
classifier = ImageClassifier()

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")
    image_bytes = await file.read()
    predictions = classifier.predict(image_bytes)
    return {"predictions": predictions}

@app.get("/health")
def health():
    return {"status": "ok"}

