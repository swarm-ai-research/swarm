"""A snapshot-friendly local wiki board."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from swarm.bridges.wiki_resampling.config import SeedEntry


@dataclass(frozen=True)
class WikiEntry:
    entry_id: int
    page: str
    author: str
    content: str
    answer: str | None


class WikiBoard:
    """Append-only board used only inside the experiment process."""

    def __init__(self, entries: Iterable[WikiEntry] = ()) -> None:
        self._entries = list(entries)

    @classmethod
    def from_seed_entries(cls, entries: Iterable[SeedEntry]) -> "WikiBoard":
        return cls(
            WikiEntry(i, e.page, e.author, e.content, e.answer)
            for i, e in enumerate(entries)
        )

    @property
    def entries(self) -> tuple[WikiEntry, ...]:
        return tuple(self._entries)

    def index(self) -> list[dict[str, object]]:
        return [
            {"entry_id": e.entry_id, "page": e.page, "author": e.author}
            for e in self._entries
        ]

    def read(self, page: str) -> list[WikiEntry]:
        return [e for e in self._entries if e.page == page]

    def write(
        self,
        *,
        page: str,
        author: str,
        content: str,
        answer: str | None,
    ) -> WikiEntry:
        entry = WikiEntry(len(self._entries), page, author, content, answer)
        self._entries.append(entry)
        return entry

    def snapshot(self) -> list[dict[str, object]]:
        return [asdict(entry) for entry in self._entries]

    @classmethod
    def restore(cls, snapshot: list[dict[str, object]]) -> "WikiBoard":
        return cls(WikiEntry(**entry) for entry in snapshot)  # type: ignore[arg-type]
