from fastapi import FastAPI, UploadFile
import shutil
import uvicorn
import os
import time
from fastapi.responses import FileResponse
from roop import core

app = FastAPI()

@app.post("/face_swap")
async def face_swap(source_image: UploadFile, target_gif: UploadFile):
    start_time = time.time()

    source_path = f"temp_{source_image.filename}"
    with open(source_path, "wb") as buffer:
        shutil.copyfileobj(source_image.file, buffer)
    target_path = f"temp_{target_gif.filename}"
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(target_gif.file, buffer)

    output_gif = core.run_replicate(source_path, target_path)

    os.remove(source_path)
    os.remove(target_path)

    finish_time = time.time()
    print(f"Execution time: {finish_time - start_time}")

    if output_gif:
        return FileResponse(output_gif)
    else:
        return output_gif

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)

