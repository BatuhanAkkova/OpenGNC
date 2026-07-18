import json
import os

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure static directory exists
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                continue

manager = ConnectionManager()

@app.get("/")
async def get():
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive commands from the dashboard if needed
            data = await websocket.receive_text()
            # For now, just print it
            print(f"Received command: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/telemetry")
async def post_telemetry(data: dict):
    """Path to push telemetry data from the simulation to the dashboard."""
    await manager.broadcast(json.dumps(data))
    return {"status": "ok"}

def run_server(port: int = 8000):
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    run_server()
