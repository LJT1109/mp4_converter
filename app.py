import uvicorn
import os
import sys
import threading
import argparse
import asyncio
import time
from contextlib import asynccontextmanager
import numpy as np
import cv2
from fastapi import FastAPI, Request, BackgroundTasks, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import shutil
import tempfile

# Cross-platform temporary file path
TEMP_NPY_PATH = os.path.join(tempfile.gettempdir(), "temp_output.npy")

# Determine Base Directory (works for PyInstaller and dev)
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure local imports work
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from converter import convert_mp4_to_npy, convert_sequence_to_npy
except ImportError:
    print("Warning: converter.py not found. Conversion features may fail.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(playback_loop())
    yield
    # Shutdown
    state.run_loop = False

app = FastAPI(lifespan=lifespan)

# Mount templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: bytes):
        for connection in self.active_connections:
            try:
                await connection.send_bytes(message)
            except Exception:
                pass

manager = ConnectionManager()

# Global State
class AppState:
    def __init__(self):
        self.video_data: Optional[np.ndarray] = None # Shape: (frames, pixels * 3)
        self.width = 0
        self.height = 0
        self.frames = 0
        self.current_frame = 0
        self.fps = 30.0
        self.playing = False
        self.last_update = time.time()
        self.grayscale = False
        self.run_loop = True
        self.filename = "output"

    def load_video(self, input_path, fps=None, grayscale=False):
        try:
            output_npy = TEMP_NPY_PATH
            # Clean up previous temp file if it exists
            if os.path.exists(output_npy):
                os.remove(output_npy)
            self.grayscale = grayscale
            
            # Store filename without extension
            base_name = os.path.basename(input_path.rstrip(os.sep))
            self.filename = os.path.splitext(base_name)[0]

            if os.path.isdir(input_path):
                # Handle Image Sequence
                convert_sequence_to_npy(input_path, output_npy, grayscale=grayscale)
                
                # Load the data
                self.video_data = np.load(output_npy)
                
                # For sequence, we infer width/height from the first frame in the array
                # Shape: (frames, pixels*3) or (frames, pixels)
                # We don't know the exact W/H unless we saved it or re-read an image.
                # Let's peek at the first image in the folder to get resolution
                valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tga')
                files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(valid_exts)])
                if files:
                    first_img = cv2.imread(os.path.join(input_path, files[0]))
                    self.height, self.width = first_img.shape[:2]
                else:
                    self.width = 100 # Default/Fallback
                    self.height = 100
                
                self.fps = float(fps) if fps is not None else 30.0
                self.frames = self.video_data.shape[0]
                
            else:
                # Handle MP4
                convert_mp4_to_npy(input_path, output_npy, grayscale=grayscale)
                
                # Load the data
                self.video_data = np.load(output_npy)
                
                # Retrieve resolution from the video directly
                cap = cv2.VideoCapture(input_path)
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                if fps is not None:
                    self.fps = float(fps)
                self.frames = self.video_data.shape[0]
                cap.release()
            
            self.current_frame = 0
            self.playing = False
            mode_str = "Grayscale" if grayscale else "RGB"
            return True, f"Loaded {self.width}x{self.height} ({mode_str}), {self.frames} frames at {self.fps} FPS."
        except Exception as e:
            return False, str(e)

state = AppState()

# Playback Loop Task
async def playback_loop():
    print("Starting Playback Loop...")
    while state.run_loop:
        if state.playing and state.frames > 0:
            now = time.time()
            if now - state.last_update > (1.0 / state.fps):
                state.current_frame = (state.current_frame + 1)
                if state.current_frame >= state.frames:
                    state.current_frame = 0 # Loop
                state.last_update = now
                
                # Broadcast frame data
                if state.video_data is not None:
                    try:
                        # Get flattened data for current frame
                        frame_data = state.video_data[state.current_frame]
                        
                        # If grayscale, repeat to create RGB structure for client
                        if state.grayscale:
                            # (Pixels,) -> (Pixels, 3) -> flattened
                            # np.repeat([1, 2], 3) -> [1, 1, 1, 2, 2, 2] which is R=G=B
                            frame_data = np.repeat(frame_data, 3)
                            
                        await manager.broadcast(frame_data.tobytes())
                    except Exception as e:
                        print(f"Error broadcasting: {e}")
                        
        await asyncio.sleep(0.001) # Small sleep to prevent busy loop

# Pydantic Models for API
class PlaybackControl(BaseModel):
    action: str # "play", "pause", "stop", "seek"
    value: Optional[int] = None

class LoadRequest(BaseModel):
    filepath: str
    fps: Optional[float] = None
    grayscale: bool = False

class SettingsRequest(BaseModel):
    fps: Optional[float] = None

# API Routes
# @app.on_event("startup")
# async def startup_event():
#     asyncio.create_task(playback_loop())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We just keep connection open, maybe handle incoming control messages later
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/load")
async def load_video(req: LoadRequest):
    # Sanitize path: remove surrounding quotes if present
    file_path = req.filepath.strip().strip("'").strip('"')
    
    if not os.path.exists(file_path):
        print(f"Error: Path not found: {file_path}")
        return JSONResponse({"status": "error", "message": f"File or directory not found: {file_path}"}, status_code=404)
        
    success, msg = state.load_video(file_path, fps=req.fps, grayscale=req.grayscale)
    if success:
        return {"status": "success", "message": msg, "meta": {
            "width": state.width,
            "height": state.height,
            "frames": state.frames,
            "fps": state.fps,
            "grayscale": state.grayscale
        }}
    else:
        return JSONResponse({"status": "error", "message": msg}, status_code=500)

@app.post("/api/settings")
async def update_settings(settings: SettingsRequest):
    if settings.fps is not None:
        state.fps = float(settings.fps)
    return {"status": "success", "fps": state.fps}

@app.post("/api/control")
async def control_playback(ctrl: PlaybackControl):
    if ctrl.action == "play":
        state.playing = True
    elif ctrl.action == "pause":
        state.playing = False
    elif ctrl.action == "stop":
        state.playing = False
        state.current_frame = 0
        # Broadcast cleared state?
    elif ctrl.action == "seek":
        if ctrl.value is not None:
            state.current_frame = max(0, min(ctrl.value, state.frames - 1))
            # Broadcast immediate frame update for seek?
            if state.video_data is not None:
                 frame_data = state.video_data[state.current_frame]
                 await manager.broadcast(frame_data.tobytes())
            
    return {"status": "success", "current_frame": state.current_frame, "playing": state.playing}

@app.get("/api/state")
async def get_state():
    return {
        "current_frame": state.current_frame,
        "playing": state.playing,
        "total_frames": state.frames,
        "width": state.width,
        "height": state.height,
        "fps": state.fps,
        "grayscale": state.grayscale
    }

@app.get("/api/download_npy")
async def download_npy():
    file_path = TEMP_NPY_PATH
    if os.path.exists(file_path):
        from fastapi.responses import FileResponse
        # Use stored filename if available, otherwise default
        fname = f"{state.filename}.npy" if state.filename else "output.npy"
        return FileResponse(file_path, filename=fname, media_type='application/octet-stream')
    return JSONResponse({"status": "error", "message": "NPY file not found"}, status_code=404)

if __name__ == "__main__":
    # open browser
    import webbrowser
    webbrowser.open("http://localhost:5050")
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="info")
