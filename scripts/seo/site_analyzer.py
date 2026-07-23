"""
Local site analyzer for swarm-ai.org.

No API keys required — works entirely off the local docs/ directory.
Produces: content inventory, internal link map, orphan pages, keyword density,
topical clusters, and actionable internal linking recommendations.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Content inventory
# ---------------------------------------------------------------------------

@dataclass
class PageInfo:
    """Metadata extracted from a single Markdown page."""
    rel_path: str
    title: str
    word_count: int
    headings: list[str]
    internal_links: list[str]   # relative paths this page links to
    external_links: list[str]
    keywords: list[str]         # top extracted terms
    has_meta_description: bool
    has_schema_markup: bool
    frontmatter: dict[str, Any] = field(default_factory=dict)


def _extract_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter if present."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    fm_block = text[3:end].strip()
    body = text[end + 3:].strip()
    fm: dict[str, Any] = {}
    for line in fm_block.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip().strip('"').strip("'")
    return fm, body


def _extract_title(text: str, fm: dict) -> str:
    """Get title from frontmatter or first H1."""
    if "title" in fm:
        return fm["title"]
    m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _extract_headings(text: str) -> list[str]:
    return re.findall(r"^#{1,6}\s+(.+)", text, re.MULTILINE)


def _extract_links(text: str) -> tuple[list[str], list[str]]:
    """Split markdown links into internal and external."""
    all_links = re.findall(r"\[([^\]]*)\]\(([^)]+)\)", text)
    internal, external = [], []
    for _label, url in all_links:
        url = url.split("#")[0].split("?")[0].strip()
        if not url:
            continue
        if url.startswith(("http://", "https://", "mailto:")):
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            path = parsed.path or ""
            is_own_domain = hostname == "swarm-ai.org" or hostname.endswith(".swarm-ai.org")
            is_own_github = hostname in ("github.com", "raw.githubusercontent.com") and path.startswith("/swarm-ai-research/")
            if is_own_domain or is_own_github:
                internal.append(url)
            else:
                external.append(url)
        elif not url.startswith("!"):  # not an image embed
            internal.append(url)
    return internal, external


_STOPWORDS = frozenset(
    "the a an and or but in on at to for of is it this that with from by as are was be "
    "been have has had do does did will would could should may might shall can not no "
    "all any each every some most more less than so if when where how what which who whom "
    "its your our their my his her we they you i me us them he she".split()
)


def _extract_keywords(text: str, top_n: int = 15) -> list[str]:
    """Naive keyword extraction: lowercased tokens minus stopwords, by frequency."""
    # strip code blocks and links
    clean = re.sub(r"```[\s\S]*?```", "", text)
    clean = re.sub(r"`[^`]+`", "", clean)
    clean = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", clean)
    clean = re.sub(r"[^a-zA-Z0-9_-]", " ", clean)
    words = [w.lower() for w in clean.split() if len(w) > 2 and w.lower() not in _STOPWORDS]
    return [w for w, _ in Counter(words).most_common(top_n)]


def analyze_page(path: Path, docs_root: Path) -> PageInfo:
    """Analyze a single Markdown file."""
    text = path.read_text(errors="replace")
    fm, body = _extract_frontmatter(text)
    internal, external = _extract_links(body)
    words = body.split()
    return PageInfo(
        rel_path=str(path.relative_to(docs_root)),
        title=_extract_title(body, fm),
        word_count=len(words),
        headings=_extract_headings(body),
        internal_links=internal,
        external_links=external,
        keywords=_extract_keywords(body),
        has_meta_description="description" in fm,
        has_schema_markup="schema" in text.lower() or "json-ld" in text.lower(),
        frontmatter=fm,
    )


# ---------------------------------------------------------------------------
# Site-wide analysis
# ---------------------------------------------------------------------------

@dataclass
class SiteAnalysis:
    pages: list[PageInfo]
    orphan_pages: list[str]           # pages no other page links to
    link_map: dict[str, list[str]]    # page -> pages it links to
    clusters: dict[str, list[str]]    # cluster name -> page paths
    linking_suggestions: list[dict]   # {from, to, reason}
    thin_content: list[str]           # pages under 300 words
    missing_meta: list[str]           # pages without meta description


def _build_link_map(pages: list[PageInfo]) -> dict[str, list[str]]:
    return {p.rel_path: p.internal_links for p in pages}


def _find_orphans(pages: list[PageInfo]) -> list[str]:
    """Pages that are never linked to by any other page."""
    all_targets: set[str] = set()
    for p in pages:
        for link in p.internal_links:
            # normalize relative links
            normalized = link.lstrip("./").rstrip("/")
            if not normalized.endswith(".md"):
                normalized += ".md" if "/" not in normalized else "/index.md"
            all_targets.add(normalized)

    all_paths = {p.rel_path for p in pages}
    # index pages are navigation, not orphans
    return sorted(p for p in all_paths - all_targets if "index.md" not in p)


def _cluster_pages(pages: list[PageInfo]) -> dict[str, list[str]]:
    """Cluster by top-level directory."""
    clusters: dict[str, list[str]] = defaultdict(list)
    for p in pages:
        parts = p.rel_path.split("/")
        cluster = parts[0] if len(parts) > 1 else "root"
        clusters[cluster].append(p.rel_path)
    return dict(clusters)


def _suggest_links(pages: list[PageInfo]) -> list[dict]:
    """
    Suggest internal links based on keyword overlap.
    If page A uses terms heavily associated with page B's title/headings,
    but A doesn't link to B, suggest the link.
    """
    # build term -> page index
    page_terms: dict[str, set[str]] = {}
    for p in pages:
        terms = set(p.keywords)
        for h in p.headings:
            terms.update(w.lower() for w in re.findall(r"\w{3,}", h))
        if p.title:
            terms.update(w.lower() for w in re.findall(r"\w{3,}", p.title))
        page_terms[p.rel_path] = terms

    suggestions: list[dict] = []
    for p in pages:
        linked_already = set(p.internal_links)
        p_kws = set(p.keywords)
        for other_path, other_terms in page_terms.items():
            if other_path == p.rel_path:
                continue
            if other_path in linked_already:
                continue
            overlap = p_kws & other_terms
            if len(overlap) >= 3:
                suggestions.append({
                    "from": p.rel_path,
                    "to": other_path,
                    "shared_terms": sorted(overlap)[:5],
                    "overlap_score": len(overlap),
                })

    # sort by overlap score descending, take top 50
    suggestions.sort(key=lambda s: s["overlap_score"], reverse=True)
    return suggestions[:50]


def analyze_site(docs_root: str | Path = "docs") -> SiteAnalysis:
    """Run full site analysis on a local docs directory."""
    root = Path(docs_root)
    if not root.is_dir():
        raise FileNotFoundError(f"docs root not found: {root}")

    md_files = sorted(root.rglob("*.md"))
    pages = [analyze_page(f, root) for f in md_files]

    return SiteAnalysis(
        pages=pages,
        orphan_pages=_find_orphans(pages),
        link_map=_build_link_map(pages),
        clusters=_cluster_pages(pages),
        linking_suggestions=_suggest_links(pages),
        thin_content=[p.rel_path for p in pages if p.word_count < 300],
        missing_meta=[p.rel_path for p in pages if not p.has_meta_description],
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def print_report(analysis: SiteAnalysis) -> None:
    """Print a human-readable summary."""
    print(f"\n{'='*60}")
    print("  SWARM-AI.ORG SITE ANALYSIS")
    print(f"{'='*60}\n")

    total_words = sum(p.word_count for p in analysis.pages)
    print(f"Total pages: {len(analysis.pages)}")
    print(f"Total words: {total_words:,}")
    print(f"Avg words/page: {total_words // max(len(analysis.pages), 1):,}")

    print("\n--- Content Clusters ---")
    for cluster, paths in sorted(analysis.clusters.items(), key=lambda x: -len(x[1])):
        print(f"  {cluster}: {len(paths)} pages")

    print(f"\n--- Thin Content (<300 words) [{len(analysis.thin_content)}] ---")
    for p in analysis.thin_content[:15]:
        page = next(pg for pg in analysis.pages if pg.rel_path == p)
        print(f"  {p} ({page.word_count} words)")
    if len(analysis.thin_content) > 15:
        print(f"  ... and {len(analysis.thin_content) - 15} more")

    print(f"\n--- Orphan Pages (no inbound links) [{len(analysis.orphan_pages)}] ---")
    for p in analysis.orphan_pages[:15]:
        print(f"  {p}")
    if len(analysis.orphan_pages) > 15:
        print(f"  ... and {len(analysis.orphan_pages) - 15} more")

    print(f"\n--- Missing Meta Description [{len(analysis.missing_meta)}] ---")
    for p in analysis.missing_meta[:10]:
        print(f"  {p}")
    if len(analysis.missing_meta) > 10:
        print(f"  ... and {len(analysis.missing_meta) - 10} more")

    print("\n--- Top Internal Linking Suggestions ---")
    for s in analysis.linking_suggestions[:10]:
        print(f"  {s['from']}")
        print(f"    -> {s['to']}")
        print(f"       shared: {', '.join(s['shared_terms'])}")

    print()


def export_csv(analysis: SiteAnalysis, output_dir: str | Path = "scripts/seo/output") -> None:
    """Export analysis to CSV files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Content inventory
    with open(out / "content_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "path", "title", "word_count", "headings_count",
            "internal_links", "external_links", "has_meta", "top_keywords",
        ])
        w.writeheader()
        for p in analysis.pages:
            w.writerow({
                "path": p.rel_path,
                "title": p.title,
                "word_count": p.word_count,
                "headings_count": len(p.headings),
                "internal_links": len(p.internal_links),
                "external_links": len(p.external_links),
                "has_meta": p.has_meta_description,
                "top_keywords": "; ".join(p.keywords[:5]),
            })

    # Link suggestions
    with open(out / "link_suggestions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["from", "to", "overlap_score", "shared_terms"])
        w.writeheader()
        for s in analysis.linking_suggestions:
            w.writerow({
                "from": s["from"],
                "to": s["to"],
                "overlap_score": s["overlap_score"],
                "shared_terms": "; ".join(s["shared_terms"]),
            })

    # Orphans
    with open(out / "orphan_pages.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path"])
        for p in analysis.orphan_pages:
            w.writerow([p])

    print(f"Exported to {out}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze swarm-ai.org docs for SEO")
    parser.add_argument("--docs", default="docs", help="Path to docs/ directory")
    parser.add_argument("--export", action="store_true", help="Export CSVs")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args()

    analysis = analyze_site(args.docs)

    if args.json:
        summary = {
            "total_pages": len(analysis.pages),
            "total_words": sum(p.word_count for p in analysis.pages),
            "orphans": len(analysis.orphan_pages),
            "thin_content": len(analysis.thin_content),
            "missing_meta": len(analysis.missing_meta),
            "clusters": {k: len(v) for k, v in analysis.clusters.items()},
            "top_suggestions": analysis.linking_suggestions[:10],
        }
        print(json.dumps(summary, indent=2))
    else:
        print_report(analysis)

    if args.export:
        export_csv(analysis)
