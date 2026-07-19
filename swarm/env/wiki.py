"""Wiki domain model and task pool for Moltipedia simulations."""

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class PageStatus(Enum):
    """Lifecycle status for a wiki page."""

    STUB = "stub"
    DRAFT = "draft"
    PUBLISHED = "published"
    CONTESTED = "contested"
    LOCKED = "locked"


class EditType(Enum):
    """Types of edits for scoring."""

    CREATE = "create"
    EDIT = "edit"
    CONTESTED_RESOLVE = "contested_resolve"
    POLICY_FIX = "policy_fix"


class PolicyViolationType(Enum):
    """Policy violation categories."""

    NEUTRALITY = "neutrality"
    SCOPE = "scope"
    SPECULATION = "speculation"
    SOURCING = "sourcing"
    NONE = "none"


@dataclass
class WikiEdit:
    """A single edit record."""

    edit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    editor_id: str = ""
    edit_type: EditType = EditType.EDIT
    delta_quality: float = 0.0
    epoch: int = 0
    step: int = 0
    note: str = ""


@dataclass
class WikiPage:
    """Represents a wiki page in the Moltipedia environment."""

    page_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    status: PageStatus = PageStatus.DRAFT
    quality_score: float = 0.3
    edit_history: List[WikiEdit] = field(default_factory=list)
    cooldown_until: int = 0
    policy_violations: List[PolicyViolationType] = field(default_factory=list)
    created_by: Optional[str] = None
    last_editor: Optional[str] = None

    def apply_edit(
        self,
        editor_id: str,
        new_content: str,
        edit_type: EditType,
        delta_quality: float,
        policy_violations: List[PolicyViolationType],
        epoch: int,
        step: int,
        note: str = "",
    ) -> None:
        """Apply an edit to the page."""
        self.content = new_content
        self.quality_score = max(0.0, min(1.0, self.quality_score + delta_quality))
        self.policy_violations = list(policy_violations)
        self.last_editor = editor_id
        self.edit_history.append(
            WikiEdit(
                editor_id=editor_id,
                edit_type=edit_type,
                delta_quality=delta_quality,
                epoch=epoch,
                step=step,
                note=note,
            )
        )

    def to_dict(self) -> Dict:
        """Serialize the page."""
        return {
            "page_id": self.page_id,
            "title": self.title,
            "content": self.content,
            "status": self.status.value,
            "quality_score": self.quality_score,
            "cooldown_until": self.cooldown_until,
            "policy_violations": [v.value for v in self.policy_violations],
            "created_by": self.created_by,
            "last_editor": self.last_editor,
        }


class WikiTaskPool:
    """Pool of wiki pages and work queues."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._pages: Dict[str, WikiPage] = {}
        self._rng = random.Random(seed)
        self._leaderboard: Dict[str, float] = {}

    @property
    def leaderboard(self) -> Dict[str, float]:
        """Return current leaderboard (agent_id -> points)."""
        return dict(self._leaderboard)

    def award_points(self, agent_id: str, points: float) -> None:
        """Award points to an agent."""
        self._leaderboard[agent_id] = self._leaderboard.get(agent_id, 0.0) + points

    def add_page(self, page: WikiPage) -> None:
        """Add a page to the pool."""
        self._pages[page.page_id] = page

    def get_page(self, page_id: str) -> Optional[WikiPage]:
        """Get page by ID."""
        return self._pages.get(page_id)

    def all_pages(self) -> List[WikiPage]:
        """Return all pages."""
        return list(self._pages.values())

    def seed_pages(self, n_pages: int, *, created_by: Optional[str] = None) -> None:
        """Seed pool with initial pages."""
        for i in range(n_pages):
            title = f"Seed Page {i + 1}"
            content = f"Stub content for {title}."
            status = PageStatus.STUB
            quality = self._rng.uniform(0.2, 0.45)
            page = WikiPage(
                title=title,
                content=content,
                status=status,
                quality_score=quality,
                created_by=created_by,
                last_editor=created_by,
            )
            self.add_page(page)

    def get_contested_pages(self, limit: int, current_step: int) -> List[WikiPage]:
        """Get contested pages that are off cooldown."""
        pages = [
            p
            for p in self._pages.values()
            if p.status == PageStatus.CONTESTED and p.cooldown_until <= current_step
        ]
        self._rng.shuffle(pages)
        return pages[:limit]

    def get_random_pages(self, limit: int, current_step: int) -> List[WikiPage]:
        """Get random pages that are off cooldown and not locked."""
        pages = [
            p
            for p in self._pages.values()
            if p.status != PageStatus.LOCKED and p.cooldown_until <= current_step
        ]
        self._rng.shuffle(pages)
        return pages[:limit]

    def get_search_pages(
        self,
        query: str,
        limit: int,
        current_step: int,
    ) -> List[WikiPage]:
        """Search pages by keyword in title/content, sorted by quality score (lowest first)."""
        if not query:
            return []
        query_l = query.lower()
        hits = [
            p
            for p in self._pages.values()
            if p.cooldown_until <= current_step
            and (query_l in p.title.lower() or query_l in p.content.lower())
        ]
        hits.sort(key=lambda p: p.quality_score)
        return hits[:limit]

    def get_low_quality_pages(self, limit: int, current_step: int) -> List[WikiPage]:
        """Get low-quality or policy-violating pages."""
        pages = [
            p
            for p in self._pages.values()
            if p.cooldown_until <= current_step
            and (p.quality_score < 0.5 or p.policy_violations)
        ]
        pages.sort(key=lambda p: p.quality_score)
        return pages[:limit]
