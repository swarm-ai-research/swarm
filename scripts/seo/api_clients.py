"""
SEO API client library for swarm-ai.org.

Free-tier clients:
  - GoogleAutocompleteClient  (no key needed)
  - SerperClient              (SERPER_API_KEY — 2,500 free searches)
  - GoogleSearchConsoleClient (GSC_SERVICE_ACCOUNT_PATH — free OAuth)
  - GoogleTrendsClient        (no key needed, via pytrends)

Paid clients (optional):
  - KeywordsEverywhereClient  (KEYWORDS_EVERYWHERE_API_KEY)
  - DataForSEOClient          (DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD)

CMS:
  - GitHubCMSClient           (GITHUB_TOKEN)

All clients are lazy — they only fail when you call a method, not on import.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


# ===========================================================================
# FREE: Google Autocomplete (no key needed)
# ===========================================================================

@dataclass
class GoogleAutocompleteClient:
    """Scrape Google Autocomplete suggestions — no API key required."""

    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def suggest(self, query: str, lang: str = "en", country: str = "us") -> list[str]:
        """Return autocomplete suggestions for a query."""
        params = {
            "q": query,
            "client": "firefox",  # returns clean JSON
            "hl": lang,
            "gl": country,
        }
        resp = self._session.get(
            "https://suggestqueries.google.com/complete/search",
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data[1] if len(data) > 1 else []

    def expand_seeds(
        self, seeds: list[str], lang: str = "en", country: str = "us", delay: float = 0.5
    ) -> list[str]:
        """Expand seed keywords via autocomplete. Appends a-z to each seed."""
        seen: set[str] = set()
        results: list[str] = []
        for seed in seeds:
            # Base query
            for suggestion in self.suggest(seed, lang=lang, country=country):
                s = suggestion.lower().strip()
                if s not in seen:
                    seen.add(s)
                    results.append(s)
            time.sleep(delay)
            # Alphabet expansion
            for letter in "abcdefghijklmnopqrstuvwxyz":
                for suggestion in self.suggest(f"{seed} {letter}", lang=lang, country=country):
                    s = suggestion.lower().strip()
                    if s not in seen:
                        seen.add(s)
                        results.append(s)
                time.sleep(delay)
        return results


# ===========================================================================
# FREE: Serper.dev (2,500 free searches)
# ===========================================================================

@dataclass
class SerperClient:
    """Serper.dev Google SERP API — 2,500 free searches."""

    api_key: str = field(default_factory=lambda: os.environ.get("SERPER_API_KEY", ""))
    base_url: str = "https://google.serper.dev"
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "SERPER_API_KEY not set. Sign up free at https://serper.dev "
                "and add it to .env."
            )

    def search(
        self, query: str, num: int = 10, gl: str = "us", hl: str = "en"
    ) -> dict:
        """Google organic search results."""
        self._require_key()
        resp = self._session.post(
            f"{self.base_url}/search",
            json={"q": query, "num": num, "gl": gl, "hl": hl},
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()

    def batch_search(
        self, queries: list[str], num: int = 10, gl: str = "us", delay: float = 0.3
    ) -> list[dict]:
        """Search multiple queries with rate limiting."""
        results = []
        for q in queries:
            results.append(self.search(q, num=num, gl=gl))
            time.sleep(delay)
        return results

    def find_ranking(self, query: str, target_domain: str = "swarm-ai.org") -> dict:
        """Check where target_domain ranks for a query."""
        data = self.search(query)
        for i, result in enumerate(data.get("organic", []), 1):
            if target_domain in result.get("link", ""):
                return {"query": query, "position": i, "url": result["link"], "title": result["title"]}
        return {"query": query, "position": None, "url": None, "title": None}


# ===========================================================================
# FREE: Google Search Console API
# ===========================================================================

@dataclass
class GoogleSearchConsoleClient:
    """Google Search Console API — free, requires service account or OAuth."""

    credentials_path: str = field(
        default_factory=lambda: os.environ.get("GSC_SERVICE_ACCOUNT_PATH", "")
    )
    site_url: str = field(
        default_factory=lambda: os.environ.get("GSC_SITE_URL", "https://swarm-ai.org")
    )
    _service: Any = field(default=None, repr=False)

    def _get_service(self) -> Any:
        if self._service is not None:
            return self._service
        if not self.credentials_path:
            raise RuntimeError(
                "GSC_SERVICE_ACCOUNT_PATH not set. Create a service account at "
                "https://console.cloud.google.com/iam-admin/serviceaccounts, "
                "download the JSON key, and add the path to .env."
            )
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError as err:
            raise RuntimeError(
                "Install GSC dependencies: pip install google-api-python-client google-auth"
            ) from err
        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
        )
        self._service = build("searchconsole", "v1", credentials=creds)
        return self._service

    def query(
        self,
        start_date: str,
        end_date: str,
        dimensions: list[str] | None = None,
        row_limit: int = 1000,
        dimension_filters: list[dict] | None = None,
    ) -> list[dict]:
        """
        Query GSC Search Analytics.

        Args:
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            dimensions: e.g. ["query", "page"]
            row_limit: max 25000
        """
        service = self._get_service()
        body: dict[str, Any] = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": dimensions or ["query"],
            "rowLimit": row_limit,
        }
        if dimension_filters:
            body["dimensionFilterGroups"] = [{"filters": dimension_filters}]
        resp = service.searchanalytics().query(siteUrl=self.site_url, body=body).execute()
        return resp.get("rows", [])

    def top_queries(self, days: int = 90, limit: int = 100) -> list[dict]:
        """Get top queries by impressions for the last N days."""
        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = self.query(start, end, dimensions=["query"], row_limit=limit)
        return sorted(rows, key=lambda r: r.get("impressions", 0), reverse=True)

    def top_pages(self, days: int = 90, limit: int = 50) -> list[dict]:
        """Get top pages by impressions for the last N days."""
        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = self.query(start, end, dimensions=["page"], row_limit=limit)
        return sorted(rows, key=lambda r: r.get("impressions", 0), reverse=True)

    def query_page_matrix(self, days: int = 90, limit: int = 500) -> list[dict]:
        """Get query × page matrix — which queries land on which pages."""
        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self.query(start, end, dimensions=["query", "page"], row_limit=limit)


# ===========================================================================
# FREE: Google Trends (pytrends — no key needed)
# ===========================================================================

@dataclass
class GoogleTrendsClient:
    """Google Trends via pytrends — no API key needed."""

    def _get_pytrends(self) -> Any:
        try:
            from pytrends.request import TrendReq
        except ImportError as err:
            raise RuntimeError("Install pytrends: pip install pytrends") from err
        return TrendReq(hl="en-US", tz=360)

    def interest_over_time(self, keywords: list[str], timeframe: str = "today 12-m") -> Any:
        """Get interest over time for up to 5 keywords."""
        pt = self._get_pytrends()
        pt.build_payload(keywords[:5], timeframe=timeframe)
        return pt.interest_over_time()

    def related_queries(self, keyword: str) -> dict:
        """Get related queries (top + rising) for a keyword."""
        pt = self._get_pytrends()
        pt.build_payload([keyword])
        return pt.related_queries()

    def suggestions(self, keyword: str) -> list[dict]:
        """Get keyword suggestions from Google Trends."""
        pt = self._get_pytrends()
        return pt.suggestions(keyword)


# ===========================================================================
# PAID: Keywords Everywhere (optional)
# ===========================================================================

@dataclass
class KeywordsEverywhereClient:
    """Thin wrapper around Keywords Everywhere v1 API."""

    api_key: str = field(default_factory=lambda: os.environ.get("KEYWORDS_EVERYWHERE_API_KEY", ""))
    base_url: str = "https://api.keywordseverywhere.com/v1"
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "KEYWORDS_EVERYWHERE_API_KEY not set. "
                "Add it to .env or pass api_key= to the constructor."
            )

    def get_volume(self, keywords: list[str], country: str = "us", currency: str = "USD") -> dict:
        self._require_key()
        resp = self._session.post(
            f"{self.base_url}/get_keyword_data",
            json={"kw": keywords, "country": country, "currency": currency},
        )
        resp.raise_for_status()
        return resp.json()

    def related_keywords(self, keyword: str, country: str = "us") -> dict:
        self._require_key()
        resp = self._session.post(
            f"{self.base_url}/get_related_keywords",
            json={"kw": [keyword], "country": country},
        )
        resp.raise_for_status()
        return resp.json()

    def people_also_search(self, keyword: str, country: str = "us") -> dict:
        self._require_key()
        resp = self._session.post(
            f"{self.base_url}/get_pasf_keywords",
            json={"kw": [keyword], "country": country},
        )
        resp.raise_for_status()
        return resp.json()


# ===========================================================================
# PAID: DataForSEO (optional)
# ===========================================================================

@dataclass
class DataForSEOClient:
    """Wrapper around DataForSEO REST API."""

    login: str = field(default_factory=lambda: os.environ.get("DATAFORSEO_LOGIN", ""))
    password: str = field(default_factory=lambda: os.environ.get("DATAFORSEO_PASSWORD", ""))
    base_url: str = "https://api.dataforseo.com/v3"
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        if self.login and self.password:
            creds = base64.b64encode(f"{self.login}:{self.password}".encode()).decode()
            self._session.headers.update({"Authorization": f"Basic {creds}"})

    def _require_creds(self) -> None:
        if not self.login or not self.password:
            raise RuntimeError(
                "DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD not set."
            )

    def _post(self, endpoint: str, payload: list[dict]) -> dict:
        self._require_creds()
        resp = self._session.post(f"{self.base_url}/{endpoint}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def serp_google(self, keywords: list[str], location: str = "United States", depth: int = 10) -> dict:
        tasks = [{"keyword": kw, "location_name": location, "language_name": "en", "depth": depth} for kw in keywords]
        return self._post("serp/google/organic/live/advanced", tasks)

    def domain_intersection(self, targets: list[str], exclude: str, limit: int = 100) -> dict:
        return self._post("backlinks/domain_intersection/live", [{"targets": targets, "exclude_targets": [exclude], "limit": limit}])


# ===========================================================================
# GitHub CMS (MkDocs publish via git)
# ===========================================================================

@dataclass
class GitHubCMSClient:
    """Publish MkDocs content by committing Markdown to the swarm repo."""

    token: str = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))
    owner: str = field(default_factory=lambda: os.environ.get("GITHUB_REPO_OWNER", "swarm-ai-research"))
    repo: str = field(default_factory=lambda: os.environ.get("GITHUB_REPO_NAME", "swarm"))
    branch: str = field(default_factory=lambda: os.environ.get("GITHUB_BRANCH", "main"))
    docs_path: str = field(default_factory=lambda: os.environ.get("MKDOCS_DOCS_PATH", "docs/"))
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        if self.token:
            self._session.headers.update({
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            })

    @property
    def api_base(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repo}"

    def get_file(self, path: str) -> dict | None:
        resp = self._session.get(f"{self.api_base}/contents/{path}", params={"ref": self.branch})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def upsert_file(self, path: str, content: str, message: str) -> dict:
        existing = self.get_file(path)
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": self.branch,
        }
        if existing:
            payload["sha"] = existing["sha"]
        resp = self._session.put(f"{self.api_base}/contents/{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def publish_page(self, rel_path: str, markdown: str, commit_msg: str | None = None) -> dict:
        full_path = f"{self.docs_path.rstrip('/')}/{rel_path}"
        msg = commit_msg or f"seo: publish {rel_path}"
        return self.upsert_file(full_path, markdown, msg)


# ===========================================================================
# Convenience factory
# ===========================================================================

def get_clients(free_only: bool = False) -> dict[str, Any]:
    """
    Return all clients. Safe to call even without API keys — clients fail lazily.

    Args:
        free_only: If True, only return free-tier clients.
    """
    clients: dict[str, Any] = {
        "autocomplete": GoogleAutocompleteClient(),
        "serper": SerperClient(),
        "gsc": GoogleSearchConsoleClient(),
        "trends": GoogleTrendsClient(),
        "cms": GitHubCMSClient(),
    }
    if not free_only:
        clients["ke"] = KeywordsEverywhereClient()
        clients["dseo"] = DataForSEOClient()
    return clients
