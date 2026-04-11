from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PoCState(Enum):
    IDLE = "IDLE"
    GENERATING = "GENERATING"
    STOPPED = "STOPPED"


@dataclass
class PoCConfig:
    """Configuration for a PoC generation round."""
    block_hash: str
    block_height: int
    public_key: str
    node_id: int = 0
    node_count: int = 1
    batch_size: int = 32
    seq_len: int = 256
    k_dim: int = 12
    callback_url: Optional[str] = None
