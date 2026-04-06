"""Configuration for VLM Navigation Brain."""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- ROS2 Topics (mini_r1_v1, from ros_gz_bridge.yaml) ---
CAMERA_TOPIC = "/r1_mini/camera/image_raw"
ODOM_TOPIC = "/r1_mini/odom"
CMD_VEL_TOPIC = "/cmd_vel"
# Also available: /r1_mini/camera/depth_image, /r1_mini/lidar, /r1_mini/imu

# --- VLM API ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "qwen/qwen3.5-397b-a17b")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-vl-72b-instruct:free")

# --- Local On-Device VLM (e.g. Qwen2.5-VL-7B via vLLM) ---
LOCAL_VLM_ENABLED = os.getenv("LOCAL_VLM_ENABLED", "false").lower() == "true"
LOCAL_VLM_BASE_URL = os.getenv("LOCAL_VLM_BASE_URL", "http://localhost:8000/v1")
LOCAL_VLM_MODEL = os.getenv("LOCAL_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
LOCAL_VLM_API_KEY = os.getenv("LOCAL_VLM_API_KEY", "not-needed")

# --- Safety Clamping ---
MAX_SPEED = 0.5
MIN_SPEED = 0.0
MAX_DURATION = 2.0
MIN_DURATION = 0.5
MAX_TURN_ANGLE = 1.0
MIN_TURN_ANGLE = 0.0

# --- Loop Timing ---
MIN_VLM_INTERVAL = 3.0  # seconds between VLM calls (rate limit protection)
SLEEP_CHUNK = 0.1  # seconds per sleep chunk (for pause flag responsiveness)

# --- Stuck Detection ---
STUCK_POSITION_THRESHOLD = 0.05  # meters
STUCK_HEADING_THRESHOLD = 5.0  # degrees
STUCK_CYCLE_COUNT = 3

# --- Revisit Detection ---
REVISIT_DISTANCE_THRESHOLD = 0.5  # meters
REVISIT_RECORD_INTERVAL = 3  # record position every N cycles

# --- Observation Memory ---
MAX_OBSERVATIONS = 5

# --- Frame Processing ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80

# --- Dashboard ---
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8765

# --- Prompt ---
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "navigation.txt")
if not os.path.exists(PROMPT_FILE):
    # Fallback: try installed share path
    PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "navigation.txt")

# --- Valid Actions ---
VALID_ACTIONS = {"forward", "left", "right", "backward", "stop"}
