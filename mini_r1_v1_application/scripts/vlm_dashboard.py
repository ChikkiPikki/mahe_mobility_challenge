#!/usr/bin/env python3
"""Web Dashboard - FastAPI + WebSocket for reasoning stream."""

import asyncio
import json
import os
import threading

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import rclpy
from rclpy.executors import MultiThreadedExecutor

import os, sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
import vlm_config as config
from vlm_brain_node import NavigationBrain, _vlmap_available, _sign_detector_available, VLMAP_ENABLED, SIGN_DETECTOR_ENABLED

# Optional hybrid modules
VLMapBuilder = None
SignDetector = None
if _vlmap_available:
    from vlmap_builder import VLMapBuilder
if _sign_detector_available:
    from sign_detector import SignDetector

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "vlm_static")), name="vlm_static")

# Global state
brain: NavigationBrain | None = None
connected_clients: set[WebSocket] = set()


async def broadcast(data: dict):
    """Send data to all connected WebSocket clients."""
    message = json.dumps(data)
    disconnected = set()
    for ws in connected_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    connected_clients -= disconnected


def on_reasoning_sync(data: dict):
    """Bridge sync callback to async broadcast."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(data), loop)


@app.get("/")
async def index():
    """Serve the dashboard HTML."""
    html_path = os.path.join(os.path.dirname(__file__), "vlm_static", "index.html")
    with open(html_path) as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Handle WebSocket connections for reasoning stream."""
    await ws.accept()
    connected_clients.add(ws)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "set_objective":
                objective = msg.get("objective", "")
                if objective and brain:
                    brain.set_objective(objective)
                    await broadcast({
                        "type": "objective_set",
                        "objective": objective,
                    })

            elif msg.get("type") == "stop":
                if brain:
                    brain.stop_navigation()
                    await broadcast({"type": "stopped"})

    except WebSocketDisconnect:
        connected_clients.discard(ws)
    except Exception:
        connected_clients.discard(ws)


def main():
    global brain

    rclpy.init()
    brain = NavigationBrain()

    # Set up reasoning callback
    brain.on_reasoning = on_reasoning_sync

    # Spin up hybrid modules alongside brain
    executor = MultiThreadedExecutor()
    executor.add_node(brain)

    if _vlmap_available and VLMAP_ENABLED and VLMapBuilder:
        brain.vlmap = VLMapBuilder()
        executor.add_node(brain.vlmap)
        print("[dashboard] VLMap spatial memory ENABLED")

    if _sign_detector_available and SIGN_DETECTOR_ENABLED and SignDetector:
        brain.sign_detector = SignDetector()
        executor.add_node(brain.sign_detector)
        print("[dashboard] Sign detector ENABLED")

    # Spin ROS2 in background
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Run brain loop in background
    brain_thread = threading.Thread(target=brain.run_loop, daemon=True)
    brain_thread.start()

    # Run FastAPI (blocks)
    try:
        uvicorn.run(
            app,
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            log_level="info",
        )
    except KeyboardInterrupt:
        pass
    finally:
        brain.shutdown()
        if brain.vlmap:
            brain.vlmap.destroy_node()
        if brain.sign_detector:
            brain.sign_detector.destroy_node()
        brain.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
