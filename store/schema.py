from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PhotoPayload:
    filename:str
    path:str
    tags: list[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "path": self.path,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height
        }

