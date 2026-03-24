import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "MiniMaxAI/MiniMax-M2.5:novita"


def read_hf_token() -> Optional[str]:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HugginFaceToken")
        or os.getenv("HuggingFaceToken")
    )
    if not token:
        return None
    return token.strip().strip('"').strip("'")
