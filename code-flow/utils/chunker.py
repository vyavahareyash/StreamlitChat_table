import tiktoken
from typing import List, Dict

def chunk_text(content: str, max_tokens: int = 1500) -> List[str]:
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(content)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(enc.decode(chunk))
    
    return chunks
