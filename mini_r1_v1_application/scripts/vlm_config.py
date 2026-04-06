"""VLM Brain configuration — API providers, model settings, tool params."""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# ── Local VLM (vLLM on host, Docker connects via --network=host) ────────
LOCAL_VLM_ENABLED = os.getenv("LOCAL_VLM_ENABLED", "true").lower() == "true"
LOCAL_VLM_BASE_URL = os.getenv("LOCAL_VLM_BASE_URL", "http://localhost:8000/v1")
LOCAL_VLM_MODEL = os.getenv("LOCAL_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
LOCAL_VLM_API_KEY = os.getenv("LOCAL_VLM_API_KEY", "not-needed")

# ── Cloud Fallback: NVIDIA NIM ──────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "qwen/qwen3.5-397b-a17b")

# ── Cloud Fallback: OpenRouter ──────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-vl-72b-instruct:free")

# ── VLM Call Settings ───────────────────────────────────────────────────
VLM_CYCLE_INTERVAL_S = 3.0   # seconds between VLM calls
VLM_MAX_TOKENS = 300
VLM_TEMPERATURE = 0.1
VLM_TIMEOUT_S = 30.0         # API call timeout

# ── Frame Processing ────────────────────────────────────────────────────
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80

# ── Observation Memory ──────────────────────────────────────────────────
MAX_OBSERVATIONS = 5

# ── Prompt ──────────────────────────────────────────────────────────────
PROMPT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts', 'navigation.txt')

# ── Valid Actions ───────────────────────────────────────────────────────
VALID_ACTIONS = {"continue", "execute_behavior", "execute_recovery", "set_speed_profile"}
