from pydantic import BaseModel
from vllm.transformers_utils.tokenizer import (
    AnyTokenizer,
    get_tokenizer,
)


from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer


class EnforcedToken(BaseModel):
    token: str
    top_tokens: List[str] = Field(default_factory=list)

    token_id: Optional[int] = Field(default=None, exclude=True)
    top_token_ids: List[int] = Field(default_factory=list, exclude=True)

    def encode(self, tokenizer: AnyTokenizer) -> List[int]:
        try:
            self.token_id = int(self.token)
            self.top_token_ids = [int(t) for t in self.top_tokens]
        except Exception as e:
            raise e


class EnforcedTokens(BaseModel):
    tokens: List[EnforcedToken]

    def encode(self, tokenizer: AnyTokenizer):
        for token in self.tokens:
            token.encode(tokenizer)

    @classmethod
    def from_content(cls, content: List[Dict[str, Any]]) -> "EnforcedTokens":
        tokens = []
        for position in content:
            token = position["token"]
            top_tokens = [x["token"] for x in position["top_logprobs"]]
            tokens.append(EnforcedToken(token=token, top_tokens=top_tokens))
        return cls(tokens=tokens)

    def get_enforced_token_ids(self) -> List[int]:
        if not self.tokens[0].token_id:
            raise ValueError("Enforced tokens are not encoded")
        return [token.token_id for token in self.tokens]
    
    
    
    