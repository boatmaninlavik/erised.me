from dataclasses import dataclass, field
from typing import Optional
import os
from pathlib import Path


def _load_dotenv():
    """Load .env file from project root if it exists."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_load_dotenv()


@dataclass
class ErisedConfig:
    model_path: str = "./ckpt"
    version: str = "3B"

    mula_device: str = "cuda:0"
    codec_device: str = "cuda:0"
    mula_dtype: str = "bfloat16"
    codec_dtype: str = "float32"
    lazy_load: bool = False  # Keep both models resident for faster pair generation

    max_audio_length_ms: int = 240_000
    topk: int = 50
    temperature: float = 1.0
    cfg_scale: float = 1.5

    llm_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    use_llm_for_tags: bool = True

    host: str = "0.0.0.0"
    port: int = 8000
    output_dir: str = "/workspace/erised_data/outputs"

    dpo_db_path: str = "/workspace/erised_data/preferences.db"
    dpo_learning_rate: float = 1e-7
    dpo_beta: float = 0.1
    dpo_epochs: int = 3
    dpo_batch_size: int = 1

    @classmethod
    def from_env(cls) -> "ErisedConfig":
        return cls(
            model_path=os.environ.get("ERISED_MODEL_PATH", "./ckpt"),
            version=os.environ.get("ERISED_MODEL_VERSION", "3B"),
            mula_device=os.environ.get("ERISED_MULA_DEVICE", "cuda:0"),
            codec_device=os.environ.get("ERISED_CODEC_DEVICE", "cuda:0"),
            lazy_load=os.environ.get("ERISED_LAZY_LOAD", "false").lower() == "true",
            llm_api_key=os.environ.get("GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            llm_model=os.environ.get("ERISED_LLM_MODEL", "gemini-2.0-flash"),
            llm_base_url=os.environ.get("ERISED_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            use_llm_for_tags=os.environ.get("ERISED_USE_LLM_TAGS", "true").lower() == "true",
            output_dir=os.environ.get("ERISED_OUTPUT_DIR", "./outputs"),
            dpo_db_path=os.environ.get("ERISED_DPO_DB", "./dpo_preferences.db"),
        )
