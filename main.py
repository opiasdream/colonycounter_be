from counter import Counter
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

app = FastAPI(title="Colony Counter API",
             description="API for counting bacterial colonies in petri dish images")
@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    try:
        # Log: Dosya bilgileri
        print(f"Uploaded file: {file.filename}, Content-Type: {file.content_type}")

        # Dosya tipini doğrula
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Dosyayı oku ve numpy array'e çevir
        image_data = await file.read()
        try:
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # Counter işlemleri
        try:
            counter = Counter(image_array=image_array, verbose=False)
            counter.detect_area_by_canny(radius=300)
            counter.crop_samples(shrinkage_ratio=0.8)
            counter.subtract_background()
            counter.detect_colonies(min_size=5, max_size=15, threshold=0.02)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")

        # Sonuç görselini oluştur
        fig, ax = plt.subplots(figsize=(10, 10))
        counter.plot_detected_colonies(ax=ax, plot="final")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()
        return StreamingResponse(buf, media_type="image/png")

    except HTTPException as e:
        print(f"Client error: {e.detail}")
        raise e
    except Exception as e:
        print(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
