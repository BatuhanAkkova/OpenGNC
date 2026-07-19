import json
import os

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "dashboard", "static")
STATIC_DIR = LEGACY_STATIC_DIR if os.path.exists(LEGACY_STATIC_DIR) else os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                continue


manager = ConnectionManager()


@app.get("/")
async def get() -> HTMLResponse:
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received command: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/telemetry")
async def post_telemetry(data: dict) -> dict[str, str]:
    await manager.broadcast(json.dumps(data))
    return {"status": "ok"}


def run_server(port: int = 8000) -> None:
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_server()
