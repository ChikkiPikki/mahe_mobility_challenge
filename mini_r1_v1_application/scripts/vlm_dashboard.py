#!/usr/bin/env python3
"""
VLM Dashboard — FastAPI + WebSocket server for real-time VLM reasoning visualization.
Subscribes to /vlm_brain/status and broadcasts to all connected web clients.

Run standalone: python3 vlm_dashboard.py
Or alongside ROS2: ros2 run mini_r1_v1_application vlm_dashboard.py
"""
import os
import sys
import json
import asyncio
import threading

# ROS2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()
connected_clients: list[WebSocket] = []
latest_status = {}


# ── WebSocket ───────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    try:
        # Send current state immediately
        if latest_status:
            await ws.send_json(latest_status)
        while True:
            data = await ws.receive_text()
            # Future: handle dashboard → brain commands here
    except WebSocketDisconnect:
        connected_clients.remove(ws)


async def broadcast(data: dict):
    for ws in connected_clients[:]:
        try:
            await ws.send_json(data)
        except Exception:
            connected_clients.remove(ws)


# ── Static files ────────────────────────────────────────────────────────

static_dir = os.path.join(os.path.dirname(__file__), 'vlm_static')

@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, 'index.html'))

if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── ROS2 Node ───────────────────────────────────────────────────────────

class DashboardBridgeNode(Node):
    def __init__(self, loop):
        super().__init__('vlm_dashboard_node')
        self._loop = loop
        self.create_subscription(
            String, '/vlm_brain/status', self._status_cb, 10)
        self.create_subscription(
            String, '/mini_r1/navigator/status', self._nav_cb, 10)
        self.get_logger().info("Dashboard bridge started.")

    def _status_cb(self, msg: String):
        global latest_status
        try:
            data = json.loads(msg.data)
            data['type'] = 'vlm_status'
            latest_status = data
            asyncio.run_coroutine_threadsafe(broadcast(data), self._loop)
        except json.JSONDecodeError:
            pass

    def _nav_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            data['type'] = 'navigator_status'
            asyncio.run_coroutine_threadsafe(broadcast(data), self._loop)
        except json.JSONDecodeError:
            pass


# ── Main ────────────────────────────────────────────────────────────────

def ros2_spin(node):
    rclpy.spin(node)


def main():
    rclpy.init()
    loop = asyncio.new_event_loop()

    node = DashboardBridgeNode(loop)
    ros_thread = threading.Thread(target=ros2_spin, args=(node,), daemon=True)
    ros_thread.start()

    config = uvicorn.Config(app, host="0.0.0.0", port=8765, loop="asyncio", log_level="warning")
    server = uvicorn.Server(config)

    asyncio.set_event_loop(loop)
    print(f"Dashboard at http://localhost:8765")
    loop.run_until_complete(server.serve())

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
