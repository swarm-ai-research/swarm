"""Platform clients for agent research archives (agentxiv, clawxiv)."""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests  # type: ignore[import-untyped]
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _parse_datetime(value: str | datetime | None) -> datetime:
    """Parse ISO timestamps, handling Zulu suffixes gracefully."""
    if isinstance(value, datetime):
        return value
    if not value:
        return datetime.now(timezone.utc)
    if isinstance(value, str) and value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)  # type: ignore[arg-type]
    except ValueError:
        return datetime.now(timezone.utc)


@dataclass
class Paper:
    """A research paper."""

    paper_id: str = ""
    title: str = ""
    abstract: str = ""
    categories: list[str] = field(default_factory=list)
    source: str = ""  # LaTeX source
    bib: str = ""  # BibTeX bibliography
    images: dict[str, str] = field(default_factory=dict)  # base64-encoded images
    authors: list[str] = field(default_factory=list)
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    changelog: str = ""
    upvotes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "categories": self.categories,
            "source": self.source,
            "bib": self.bib,
            "images": self.images,
            "authors": self.authors,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "changelog": self.changelog,
            "upvotes": self.upvotes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        """Deserialize from dictionary."""
        files = data.get("files", {}) if isinstance(data.get("files"), dict) else {}
        source = data.get("source") or files.get("source", "")
        bib = data.get("bib") or files.get("bib", "")
        images_raw = data.get("images") or files.get("images", {})
        images = images_raw if isinstance(images_raw, dict) else {}
        authors_raw = data.get("authors", [])
        if isinstance(authors_raw, list) and authors_raw:
            if isinstance(authors_raw[0], dict):
                authors = [a.get("name", "") for a in authors_raw if a.get("name")]
            else:
                authors = [str(a) for a in authors_raw]
        else:
            authors = []
        return cls(
            paper_id=data.get("paper_id", data.get("id", "")),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            categories=data.get("categories", []),
            source=source,
            bib=bib,
            images=images,
            authors=authors,
            version=data.get("version", 1),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
            changelog=data.get("changelog", ""),
            upvotes=data.get("upvotes", 0),
        )

    def content_hash(self) -> str:
        """Compute hash of paper content for verification."""
        content = f"{self.title}|{self.abstract}|{self.source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class SearchResult:
    """Search results from a platform."""

    papers: list[Paper] = field(default_factory=list)
    total_count: int = 0
    query: str = ""


@dataclass
class SubmissionResult:
    """Result of submitting a paper."""

    success: bool = False
    paper_id: str = ""
    message: str = ""
    version: int = 1


def _error_detail(exc: requests.RequestException) -> str:
    """Extract error detail including response body when available."""
    msg = str(exc)
    if hasattr(exc, "response") and exc.response is not None:
        try:
            body = exc.response.text[:500]
            if body:
                msg = f"{msg} | {body}"
        except Exception:
            pass
    return msg


def _is_retryable(exc: BaseException) -> bool:
    """Check if an HTTP error is retryable (429 or 5xx)."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status_code = int(exc.response.status_code)
        return status_code == 429 or status_code >= 500  # type: ignore[no-any-return]
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))  # type: ignore[no-any-return]


class PlatformClient:
    """Base class for research platform clients.

    Subclasses only need to set base_url, env_var_name, and optionally auth_header.
    All HTTP methods are implemented here with retry logic.
    """

    base_url: str = ""
    env_var_name: str = ""
    auth_header: str = "Authorization"  # Override in subclass if needed

    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get(self.env_var_name)
        self.api_key = api_key
        self._session = requests.Session()
        if api_key:
            if self.auth_header == "X-API-Key":
                self._session.headers["X-API-Key"] = api_key
            else:
                self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers["Content-Type"] = "application/json"

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an HTTP request with retry on transient errors."""
        response = self._session.request(method, url, **kwargs)
        if response.status_code == 429 or response.status_code >= 500:
            logger.warning(
                "Retryable HTTP %d from %s %s",
                response.status_code,
                method,
                url,
            )
            response.raise_for_status()
        return response

    def search(self, query: str, *, limit: int = 20, **kwargs: Any) -> SearchResult:
        """Search for papers."""
        try:
            response = self._request(
                "POST",
                f"{self.base_url}/search",
                json={"query": query, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            papers = [
                Paper.from_dict(p) for p in data.get("papers", data.get("results", []))
            ]
            return SearchResult(
                papers=papers,
                total_count=data.get("total", len(papers)),
                query=query,
            )
        except requests.RequestException as e:
            logger.warning("Search failed for %s on %s: %s", query, self.base_url, e)
            return SearchResult(papers=[], total_count=0, query=query)

    def get_paper(self, paper_id: str) -> Paper | None:
        """Retrieve a specific paper."""
        try:
            response = self._request("GET", f"{self.base_url}/papers/{paper_id}")
            response.raise_for_status()
            return Paper.from_dict(response.json())
        except requests.RequestException as e:
            logger.warning("Get paper %s failed on %s: %s", paper_id, self.base_url, e)
            return None

    def submit(self, paper: Paper) -> SubmissionResult:
        """Submit a new paper."""
        try:
            response = self._request(
                "POST",
                f"{self.base_url}/papers",
                json={
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "source": paper.source,
                },
            )
            response.raise_for_status()
            data = response.json()
            return SubmissionResult(
                success=True,
                paper_id=data.get("paper_id", data.get("id", "")),
                message="Paper submitted successfully",
                version=1,
            )
        except requests.RequestException as e:
            logger.warning("Submit failed on %s: %s", self.base_url, e)
            return SubmissionResult(success=False, message=_error_detail(e))

    def update(self, paper_id: str, paper: Paper) -> SubmissionResult:
        """Update an existing paper."""
        try:
            response = self._request(
                "PUT",
                f"{self.base_url}/papers/{paper_id}",
                json={
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "source": paper.source,
                    "changelog": paper.changelog,
                },
            )
            response.raise_for_status()
            data = response.json()
            return SubmissionResult(
                success=True,
                paper_id=paper_id,
                message="Paper updated successfully",
                version=data.get("version", paper.version + 1),
            )
        except requests.RequestException as e:
            logger.warning("Update %s failed on %s: %s", paper_id, self.base_url, e)
            return SubmissionResult(success=False, message=_error_detail(e))

    def upvote(self, paper_id: str) -> bool:
        """Upvote a paper."""
        try:
            response = self._request(
                "POST", f"{self.base_url}/papers/{paper_id}/upvote"
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning("Upvote %s failed on %s: %s", paper_id, self.base_url, e)
            return False

    def register(self, name: str, affiliation: str) -> dict[str, str]:
        """Register a new author account."""
        response = self._request(
            "POST",
            f"{self.base_url}/register",
            json={"name": name, "affiliation": affiliation},
        )
        response.raise_for_status()
        result: dict[str, str] = response.json()  # type: ignore[assignment]
        return result


class AgentxivClient(PlatformClient):
    """Client for agentxiv.org API.

    Note: Uses Markdown content format, not LaTeX.
    All tool endpoints use POST to /api/v1/tools/*.
    Requires human verification before submission access.
    """

    base_url = "https://agentxiv.org/api/v1"
    env_var_name = "AGENTXIV_API_KEY"

    def register(self, name: str, affiliation: str) -> dict[str, str]:
        """Register an agent account.

        Returns claim_url for human verification.
        """
        response = self._request(
            "POST",
            f"{self.base_url}/agents/register",
            json={"name": name, "affiliation": affiliation},
        )
        response.raise_for_status()
        result: dict[str, str] = response.json()
        return result

    def get_status(self) -> dict[str, Any]:
        """Check agent verification status."""
        response = self._request("GET", f"{self.base_url}/agents/status")
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def search(self, query: str, *, limit: int = 20, **kwargs: Any) -> SearchResult:
        """Search for papers."""
        try:
            response = self._request(
                "POST",
                f"{self.base_url}/tools/search",
                json={"query": query, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            papers = [
                Paper.from_dict(p) for p in data.get("papers", data.get("results", []))
            ]
            return SearchResult(
                papers=papers,
                total_count=data.get("total", len(papers)),
                query=query,
            )
        except requests.RequestException as e:
            logger.warning("Search failed for %s on %s: %s", query, self.base_url, e)
            return SearchResult(papers=[], total_count=0, query=query)

    def get_paper(self, paper_id: str) -> Paper | None:
        """Retrieve a specific paper."""
        try:
            response = self._request(
                "POST",
                f"{self.base_url}/tools/read",
                json={"paper_id": paper_id},
            )
            response.raise_for_status()
            return Paper.from_dict(response.json())
        except requests.RequestException as e:
            logger.warning("Get paper %s failed on %s: %s", paper_id, self.base_url, e)
            return None

    def submit(self, paper: Paper) -> SubmissionResult:
        """Submit a new paper (Markdown format)."""
        try:
            response = self._request(
                "POST",
                f"{self.base_url}/tools/submit",
                json={
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "content": paper.source,  # agentxiv uses Markdown content
                    "category": paper.categories[0]
                    if paper.categories
                    else "multi-agent",
                },
            )
            response.raise_for_status()
            data = response.json()
            return SubmissionResult(
                success=True,
                paper_id=data.get("paper_id", data.get("id", "")),
                message="Paper submitted successfully",
                version=1,
            )
        except requests.RequestException as e:
            logger.warning("Submit failed on %s: %s", self.base_url, e)
            return SubmissionResult(success=False, message=_error_detail(e))

    def update(self, paper_id: str, paper: Paper) -> SubmissionResult:
        """Revise an existing paper."""
        try:
            response = self._request(
                "POST",
                f"{self.base_url}/tools/revise",
                json={
                    "paper_id": paper_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "content": paper.source,
                    "changelog": paper.changelog or "Updated",
                },
            )
            response.raise_for_status()
            data = response.json()
            return SubmissionResult(
                success=True,
                paper_id=paper_id,
                message="Paper revised successfully",
                version=data.get("version", paper.version + 1),
            )
        except requests.RequestException as e:
            logger.warning("Revise %s failed on %s: %s", paper_id, self.base_url, e)
            return SubmissionResult(success=False, message=_error_detail(e))

    def get_categories(self) -> list[str]:
        """Get available paper categories."""
        try:
            response = self._request(
                "POST",
                f"{self.base_url}/tools/categories",
                json={},
            )
            response.raise_for_status()
            data = response.json()
            categories: list[str] = data.get("categories", [])
            return categories
        except requests.RequestException as e:
            logger.warning("Get categories failed on %s: %s", self.base_url, e)
            return []

    def submit_review(
        self,
        paper_id: str,
        rating: int,
        comment: str,
        reply_to: str | None = None,
    ) -> bool:
        """Submit a review for a paper."""
        try:
            payload: dict[str, Any] = {
                "paper_id": paper_id,
                "rating": rating,
                "comment": comment,
            }
            if reply_to:
                payload["reply_to"] = reply_to

            response = self._request(
                "POST",
                f"{self.base_url}/tools/review",
                json=payload,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning("Review %s failed on %s: %s", paper_id, self.base_url, e)
            return False

    def get_reviews(self, paper_id: str) -> list[dict[str, Any]]:
        """Get reviews for a paper."""
        try:
            response = self._request(
                "POST",
                f"{self.base_url}/tools/reviews",
                json={"paper_id": paper_id},
            )
            response.raise_for_status()
            data = response.json()
            reviews: list[dict[str, Any]] = data.get("reviews", [])
            return reviews
        except requests.RequestException as e:
            logger.warning("Get reviews for %s failed: %s", paper_id, e)
            return []


class ClawxivClient(PlatformClient):
    """Client for clawxiv.org API.

    Note: Must use www.clawxiv.org and X-API-Key header.
    """

    base_url = "https://www.clawxiv.org/api/v1"
    env_var_name = "CLAWXIV_API_KEY"
    auth_header = "X-API-Key"

    def register(self, name: str, description: str) -> dict[str, str]:
        """Register a new bot account."""
        response = self._request(
            "POST",
            f"{self.base_url}/register",
            json={"name": name, "description": description},
        )
        response.raise_for_status()
        result: dict[str, str] = response.json()  # type: ignore[assignment]
        return result

    def submit(self, paper: Paper) -> SubmissionResult:
        """Submit a new paper (LaTeX)."""
        try:
            files: dict[str, Any] = {"source": paper.source}
            if paper.bib:
                files["bib"] = paper.bib
            if paper.images:
                files["images"] = paper.images
            response = self._request(
                "POST",
                f"{self.base_url}/papers",
                json={
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "files": files,
                },
            )
            response.raise_for_status()
            data = response.json()
            return SubmissionResult(
                success=True,
                paper_id=data.get("paper_id", data.get("id", "")),
                message="Paper submitted successfully",
                version=1,
            )
        except requests.RequestException as e:
            logger.warning("Submit failed on %s: %s", self.base_url, e)
            return SubmissionResult(success=False, message=_error_detail(e))

    def update(self, paper_id: str, paper: Paper) -> SubmissionResult:
        """Update an existing paper (single-author)."""
        try:
            files: dict[str, Any] = {"source": paper.source}
            if paper.bib:
                files["bib"] = paper.bib
            if paper.images:
                files["images"] = paper.images
            response = self._request(
                "PUT",
                f"{self.base_url}/papers/{paper_id}",
                json={
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "files": files,
                },
            )
            response.raise_for_status()
            data = response.json()
            return SubmissionResult(
                success=True,
                paper_id=paper_id,
                message="Paper updated successfully",
                version=data.get("version", paper.version + 1),
            )
        except requests.RequestException as e:
            logger.warning("Update %s failed on %s: %s", paper_id, self.base_url, e)
            return SubmissionResult(success=False, message=_error_detail(e))

    def search(
        self,
        query: str,
        *,
        title: str | None = None,
        author: str | None = None,
        abstract: str | None = None,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        sort_by: str | None = None,
        sort_order: str | None = None,
        page: int | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> SearchResult:
        """Search for papers.

        ClawXiv search uses GET with query parameters.
        """
        try:
            params: dict[str, Any] = {"query": query, "limit": limit}
            if title:
                params["title"] = title
            if author:
                params["author"] = author
            if abstract:
                params["abstract"] = abstract
            if category:
                params["category"] = category
            if date_from:
                params["date_from"] = date_from
            if date_to:
                params["date_to"] = date_to
            if sort_by:
                params["sort_by"] = sort_by
            if sort_order:
                params["sort_order"] = sort_order
            if page is not None:
                params["page"] = page

            response = self._request(
                "GET",
                f"{self.base_url}/search",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 405:
                logger.warning(
                    "Search failed for %s on %s: %s", query, self.base_url, e
                )
                return SearchResult(papers=[], total_count=0, query=query)
            # Retry with POST if GET is not supported (405 Method Not Allowed)
            try:
                response = self._request(
                    "POST",
                    f"{self.base_url}/search",
                    json={"query": query, "limit": limit},
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as e2:
                logger.warning(
                    "Search failed for %s on %s: %s", query, self.base_url, e2
                )
                return SearchResult(papers=[], total_count=0, query=query)
        except requests.RequestException as e:
            logger.warning("Search failed for %s on %s: %s", query, self.base_url, e)
            return SearchResult(papers=[], total_count=0, query=query)

        papers = [
            Paper.from_dict(p) for p in data.get("papers", data.get("results", []))
        ]
        return SearchResult(
            papers=papers,
            total_count=data.get("total", len(papers)),
            query=query,
        )


class AgentRxivClient(PlatformClient):
    """Client for AgentRxiv local preprint server.

    AgentRxiv is a local Flask-based server from Agent Laboratory
    that enables autonomous agents to share research papers.
    Uses semantic search with sentence transformers.

    Note: Unlike other platforms, AgentRxiv:
    - Runs locally (default: http://127.0.0.1:5000)
    - Does not require authentication
    - Accepts PDF uploads rather than LaTeX source
    - Uses GET for search instead of POST
    """

    base_url = "http://127.0.0.1:5000"
    env_var_name = "AGENTRXIV_URL"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize AgentRxiv client.

        Args:
            api_key: Not used (kept for interface compatibility).
            base_url: Override the default server URL.
        """
        # AgentRxiv doesn't use auth, but we follow the interface
        super().__init__(api_key=api_key)
        if base_url:
            self.base_url = base_url
        elif os.environ.get(self.env_var_name):
            self.base_url = os.environ[self.env_var_name]

    def health_check(self) -> bool:
        """Check if the AgentRxiv server is running."""
        try:
            response = self._session.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def search(self, query: str, *, limit: int = 5, **kwargs) -> SearchResult:
        """Search papers by semantic similarity.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            SearchResult with matching papers.
        """
        try:
            response = self._request(
                "GET",
                f"{self.base_url}/api/search",
                params={"q": query},
            )
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data[:limit]:
                filename = item.get("filename", "")
                title = filename.replace(".pdf", "").replace("_", " ")
                text = item.get("text", "")

                papers.append(Paper(
                    paper_id=f"agentrxiv:{item.get('id', 0)}",
                    title=title,
                    abstract=text[:500] if text else "",
                    source=text,
                    categories=["agentrxiv"],
                ))

            return SearchResult(
                papers=papers,
                total_count=len(data),
                query=query,
            )
        except requests.RequestException as e:
            logger.warning("AgentRxiv search failed: %s", e)
            return SearchResult(papers=[], total_count=0, query=query)

    def get_paper(self, paper_id: str) -> Paper | None:
        """Retrieve a paper by ID."""
        # Extract numeric ID
        if isinstance(paper_id, str) and paper_id.startswith("agentrxiv:"):
            paper_id = paper_id.split(":")[1]

        try:
            response = self._request("GET", f"{self.base_url}/view/{paper_id}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return Paper(paper_id=f"agentrxiv:{paper_id}")
        except requests.RequestException as e:
            logger.warning("Get paper %s failed: %s", paper_id, e)
            return None

    def submit(self, paper: Paper, pdf_path: str | None = None) -> SubmissionResult:
        """Submit a paper (PDF) to AgentRxiv.

        Args:
            paper: Paper metadata.
            pdf_path: Path to PDF file. Required for AgentRxiv.

        Returns:
            SubmissionResult indicating success or failure.
        """
        if not pdf_path:
            return SubmissionResult(
                success=False,
                message="AgentRxiv requires pdf_path for submission",
            )

        try:
            with open(pdf_path, "rb") as f:
                filename = f"{paper.title}.pdf" if paper.title else "paper.pdf"
                files = {"file": (filename, f, "application/pdf")}
                response = self._session.post(
                    f"{self.base_url}/upload",
                    files=files,
                    timeout=60,
                )

            if response.status_code == 200:
                return SubmissionResult(
                    success=True,
                    paper_id=f"agentrxiv:{paper.title}",
                    message="Paper uploaded successfully",
                )
            else:
                return SubmissionResult(
                    success=False,
                    message=f"Upload failed: {response.text}",
                )
        except Exception as e:
            logger.error("AgentRxiv submit failed: %s", e)
            if isinstance(e, requests.RequestException):
                return SubmissionResult(success=False, message=_error_detail(e))
            return SubmissionResult(success=False, message=str(e))

    def trigger_update(self) -> bool:
        """Trigger server to process new uploads."""
        try:
            response = self._request("GET", f"{self.base_url}/update")
            return bool(response.status_code == 200)
        except requests.RequestException:
            return False

    def submit_review(
        self,
        paper_id: str,
        rating: int,
        comment: str,
        reviewer_id: str = "",
    ) -> bool:
        """Submit a review for a paper.

        Stub for interface parity with AgentxivClient.
        Actual storage is handled by the bridge layer.
        """
        logger.info(
            "AgentRxivClient.submit_review: paper_id=%s rating=%d reviewer=%s",
            paper_id,
            rating,
            reviewer_id,
        )
        return True

    def get_reviews(self, paper_id: str) -> list[dict[str, Any]]:
        """Get reviews for a paper.

        Stub for interface parity with AgentxivClient.
        Actual retrieval is handled by the bridge layer.
        """
        return []


def get_client(platform: str, api_key: str | None = None) -> PlatformClient:
    """Factory function to get a platform client."""
    clients: dict[str, type[PlatformClient]] = {
        "agentxiv": AgentxivClient,
        "clawxiv": ClawxivClient,
        "agentrxiv": AgentRxivClient,
    }
    if platform not in clients:
        raise ValueError(
            f"Unknown platform: {platform}. Available: {list(clients.keys())}"
        )
    return clients[platform](api_key=api_key)
