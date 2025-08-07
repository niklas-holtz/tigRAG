from abc import ABC, abstractmethod


class ChunkStorage(ABC):
    @abstractmethod
    def insert_chunk(self, from_idx: int, to_idx: int, text: str, keywords: list, embedding: list | None):
        pass

    @abstractmethod
    def close(self):
        pass
