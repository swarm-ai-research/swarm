"""
Auto-linker for swarm-ai.org docs.

Scans all Markdown pages, finds natural anchor text opportunities,
and inserts internal links. Also appends a "Related pages" section
for high-overlap pages that lack inline linking opportunities.

No API keys required.

Usage:
    python scripts/seo/auto_linker.py --docs docs --dry-run     # preview changes
    python scripts/seo/auto_linker.py --docs docs --apply        # write changes
    python scripts/seo/auto_linker.py --docs docs --apply --related  # also add Related sections
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from site_analyzer import PageInfo, analyze_site

# ---------------------------------------------------------------------------
# Link target registry
# ---------------------------------------------------------------------------

@dataclass
class LinkTarget:
    """A page that other pages should link to."""
    rel_path: str
    title: str
    anchor_phrases: list[str]  # phrases that should become links to this page


def _build_targets(pages: list[PageInfo]) -> list[LinkTarget]:
    """
    Build a registry of link targets with their anchor phrases.
    Anchor phrases come from: title, H2 headings, top keywords.
    """
    targets: list[LinkTarget] = []
    for p in pages:
        if not p.title:
            continue
        phrases: list[str] = []

        # Title-derived phrases (most valuable)
        title_clean = re.sub(r"[^\w\s-]", "", p.title).strip()
        if len(title_clean.split()) <= 5 and len(title_clean) > 5:
            phrases.append(title_clean)

        # Key multi-word concepts from headings
        for h in p.headings:
            h_clean = re.sub(r"^#+\s*", "", h).strip()
            h_clean = re.sub(r"[^\w\s-]", "", h_clean).strip()
            words = h_clean.split()
            if 2 <= len(words) <= 4 and len(h_clean) > 5:
                phrases.append(h_clean)

        # Known domain terms that map to this page (based on path)
        path_terms = _path_to_terms(p.rel_path)
        phrases.extend(path_terms)

        # Deduplicate, case-insensitive
        seen: set[str] = set()
        unique: list[str] = []
        for ph in phrases:
            key = ph.lower()
            if key not in seen and len(key) > 4:
                seen.add(key)
                unique.append(ph)

        if unique:
            targets.append(LinkTarget(
                rel_path=p.rel_path,
                title=p.title,
                anchor_phrases=unique[:8],  # cap per page
            ))

    return targets


# Domain-specific term mappings based on file path
_PATH_TERM_MAP: dict[str, list[str]] = {
    "concepts/metrics": ["soft metrics", "toxicity rate", "quality gap", "conditional loss"],
    "concepts/governance": ["governance mechanisms", "transaction tax", "reputation decay", "circuit breaker"],
    "concepts/soft-labels": ["soft labels", "probabilistic labels", "soft probabilistic"],
    "concepts/emergence": ["emergent behavior", "emergent risk", "emergence"],
    "concepts/recursive-research": ["recursive research", "recursive agent"],
    "concepts/time-horizons": ["time horizons", "temporal dynamics"],
    "blog/purity-paradox": ["purity paradox"],
    "research/theory": ["distributional safety", "distributional agi safety"],
    "research/reflexivity": ["reflexivity", "reflexive dynamics"],
    "guides/red-teaming": ["red teaming", "red team"],
    "guides/llm-agents": ["LLM agents", "language model agents"],
    "guides/parameter-sweeps": ["parameter sweep", "parameter sweeps"],
    "guides/scenarios": ["scenario configuration", "scenario yaml"],
    "getting-started/installation": ["pip install", "installation"],
    "getting-started/quickstart": ["quickstart", "getting started"],
    "bridges/concordia": ["Concordia", "concordia integration"],
    "bridges/prime_intellect": ["Prime Intellect"],
}


def _path_to_terms(rel_path: str) -> list[str]:
    """Map a file path to known domain terms."""
    key = rel_path.replace(".md", "").rstrip("/")
    return _PATH_TERM_MAP.get(key, [])


# ---------------------------------------------------------------------------
# Link insertion engine
# ---------------------------------------------------------------------------

# Regions where we should NOT insert links
_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`]+`")
_EXISTING_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
_HEADING = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
_FRONTMATTER = re.compile(r"^---[\s\S]*?---\n", re.MULTILINE)
_HTML_TAG = re.compile(r"<[^>]+>")
_TABLE_ROW = re.compile(r"^\|.*\|$", re.MULTILINE)


def _build_no_link_spans(text: str) -> list[tuple[int, int]]:
    """Build a list of character spans where links should not be inserted."""
    no_link_spans: list[tuple[int, int]] = []
    for pattern in [_CODE_BLOCK, _INLINE_CODE, _EXISTING_LINK, _HEADING, _FRONTMATTER, _HTML_TAG, _TABLE_ROW]:
        for m in pattern.finditer(text):
            no_link_spans.append((m.start(), m.end()))
    return no_link_spans


def _compute_rel_link(from_path: str, to_path: str) -> str:
    """Compute a relative link from one doc to another."""
    from_parts = Path(from_path).parent.parts
    to_parts = Path(to_path).parts

    # find common prefix
    common = 0
    for a, b in zip(from_parts, to_parts, strict=False):
        if a == b:
            common += 1
        else:
            break

    ups = len(from_parts) - common
    rel = [".."] * ups + list(to_parts[common:])
    return "/".join(rel)


@dataclass
class LinkInsertion:
    """A single link insertion to be made."""
    file_path: str
    anchor_text: str
    target_path: str
    line_number: int
    context: str  # surrounding text for preview


_TOO_GENERIC = frozenset({
    "metrics", "governance", "research", "agents", "agent", "safety",
    "ai safety", "for ai safety", "overview", "introduction", "guide",
    "getting started", "installation", "quickstart", "concepts", "index",
    "conclusion", "summary", "results", "analysis", "discussion",
})


def _find_insertion_points(
    text: str,
    from_path: str,
    targets: list[LinkTarget],
    already_linked: set[str],
    max_links_per_page: int = 5,
) -> list[LinkInsertion]:
    """
    Find natural places to insert links in a page's text.
    Rules:
    - Only link the FIRST occurrence of each phrase
    - Each phrase can only link to ONE target (best match wins)
    - Never link inside code blocks, existing links, headings, tables, or HTML
    - Max N new links per page to avoid over-optimization
    - Don't link to self
    - Skip overly generic single/two-word phrases
    """
    insertions: list[LinkInsertion] = []

    # Build a mask of "no-link zones"
    no_link_spans = _build_no_link_spans(text)

    def _in_no_link_zone(start: int, end: int) -> bool:
        return any(s <= start < e or s < end <= e for s, e in no_link_spans)

    linked_targets: set[str] = set()
    used_phrases: set[str] = set()  # prevent same phrase -> multiple targets

    # Sort targets: prefer concept/guide/api pages over blog/papers, then longer phrases
    def _target_priority(t: LinkTarget) -> tuple[int, int]:
        path = t.rel_path
        # Canonical pages first (concepts, guides, api, getting-started) — but not index pages
        is_index = path.endswith("index.md")
        if any(path.startswith(p) for p in ("concepts/", "guides/", "api/", "getting-started/")):
            tier = 0 if not is_index else 2
        elif any(path.startswith(p) for p in ("research/", "bridges/")):
            tier = 1
        elif path.startswith("blog/"):
            tier = 3
        else:
            tier = 2
        max_phrase_len = max((len(p) for p in t.anchor_phrases), default=0)
        return (tier, -max_phrase_len)

    scored_targets: list[tuple[tuple[int, int], LinkTarget]] = []
    for t in targets:
        scored_targets.append((_target_priority(t), t))
    scored_targets.sort(key=lambda x: x[0])

    for _, target in scored_targets:
        if target.rel_path == from_path:
            continue
        if target.rel_path in already_linked:
            continue
        if target.rel_path in linked_targets:
            continue
        if len(insertions) >= max_links_per_page:
            break

        for phrase in target.anchor_phrases:
            phrase_lower = phrase.lower().strip()

            # Skip generic phrases
            if phrase_lower in _TOO_GENERIC:
                continue

            # Skip very short phrases (< 6 chars) — too likely to false-match
            if len(phrase_lower) < 6:
                continue

            # Skip if this exact phrase already used for another target
            if phrase_lower in used_phrases:
                continue

            # Require word boundaries to avoid partial-word matches
            pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
            match = None
            for m in pattern.finditer(text):
                if not _in_no_link_zone(m.start(), m.end()):
                    match = m
                    break

            if not match:
                continue

            # Compute line number
            line_num = text[:match.start()].count("\n") + 1

            # Get context (the line containing the match)
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = len(text)
            context_line = text[line_start:line_end].strip()

            insertions.append(LinkInsertion(
                file_path=from_path,
                anchor_text=match.group(),
                target_path=target.rel_path,
                line_number=line_num,
                context=context_line,
            ))
            linked_targets.add(target.rel_path)
            used_phrases.add(phrase_lower)
            break  # one link per target per page

    return insertions


def _apply_insertions(text: str, insertions: list[LinkInsertion], from_path: str) -> str:
    """Apply link insertions to page text, working from end to start to preserve offsets."""
    if not insertions:
        return text

    # Re-find each insertion point and apply from end to start
    changes: list[tuple[int, int, str]] = []

    no_link_spans = _build_no_link_spans(text)

    used_targets: set[str] = set()

    for ins in insertions:
        if ins.target_path in used_targets:
            continue

        pattern = re.compile(re.escape(ins.anchor_text), re.IGNORECASE)
        for m in pattern.finditer(text):
            if any(s <= m.start() < e or s < m.end() <= e for s, e in no_link_spans):
                continue
            # Check this span hasn't been claimed by a prior insertion
            if any(s <= m.start() < e for s, e, _ in changes):
                continue

            rel_link = _compute_rel_link(from_path, ins.target_path)
            replacement = f"[{m.group()}]({rel_link})"
            changes.append((m.start(), m.end(), replacement))
            used_targets.add(ins.target_path)
            # Mark this range as a no-link zone for subsequent insertions
            no_link_spans.append((m.start(), m.start() + len(replacement)))
            break

    # Sort by position descending and apply
    changes.sort(key=lambda c: c[0], reverse=True)
    for start, end, replacement in changes:
        text = text[:start] + replacement + text[end:]

    return text


# ---------------------------------------------------------------------------
# Related pages section
# ---------------------------------------------------------------------------

def _build_related_section(
    from_path: str,
    suggestions: list[dict],
    pages: list[PageInfo],
    max_related: int = 5,
) -> str | None:
    """Build a '## Related Pages' markdown section for a page."""
    related = [s for s in suggestions if s["from"] == from_path]
    if not related:
        return None

    title_map = {p.rel_path: p.title for p in pages}

    lines = ["\n\n---\n\n## Related Pages\n"]
    for s in related[:max_related]:
        target = s["to"]
        title = title_map.get(target, target)
        rel_link = _compute_rel_link(from_path, target)
        terms = ", ".join(s["shared_terms"][:3])
        lines.append(f"- [{title}]({rel_link}) — {terms}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@dataclass
class AutoLinkResult:
    """Result of the auto-linking pass."""
    total_pages: int
    pages_modified: int
    links_inserted: int
    related_sections_added: int
    insertions: list[LinkInsertion]
    errors: list[str]


def auto_link(
    docs_root: str | Path = "docs",
    dry_run: bool = True,
    add_related: bool = False,
    max_links_per_page: int = 5,
    skip_patterns: list[str] | None = None,
) -> AutoLinkResult:
    """
    Run the auto-linker across all docs.

    Args:
        docs_root: Path to docs/ directory
        dry_run: If True, only preview changes; don't write files
        add_related: If True, append Related Pages sections
        max_links_per_page: Max new inline links per page
        skip_patterns: Glob patterns to skip (e.g. ["index.md"])
    """
    root = Path(docs_root)
    analysis = analyze_site(root)
    targets = _build_targets(analysis.pages)

    skip = set(skip_patterns or [])
    # Always skip the homepage (custom HTML layout)
    skip.add("index.md")

    all_insertions: list[LinkInsertion] = []
    pages_modified = 0
    related_added = 0
    errors: list[str] = []

    for page in analysis.pages:
        if page.rel_path in skip:
            continue
        if any(page.rel_path.endswith(s) for s in skip):
            continue

        file_path = root / page.rel_path
        try:
            text = file_path.read_text(errors="replace")
        except Exception as e:
            errors.append(f"Read error {page.rel_path}: {e}")
            continue

        already_linked = set(page.internal_links)
        insertions = _find_insertion_points(
            text, page.rel_path, targets, already_linked, max_links_per_page
        )

        if not insertions and not add_related:
            continue

        new_text = _apply_insertions(text, insertions, page.rel_path)
        all_insertions.extend(insertions)

        # Related pages section
        related_section = None
        if add_related and "## Related Pages" not in new_text:
            related_section = _build_related_section(
                page.rel_path, analysis.linking_suggestions, analysis.pages
            )
            if related_section:
                # Strip trailing whitespace and add the section
                new_text = new_text.rstrip() + related_section
                related_added += 1

        if new_text != text:
            pages_modified += 1
            if not dry_run:
                file_path.write_text(new_text)

    return AutoLinkResult(
        total_pages=len(analysis.pages),
        pages_modified=pages_modified,
        links_inserted=len(all_insertions),
        related_sections_added=related_added,
        insertions=all_insertions,
        errors=errors,
    )


def print_preview(result: AutoLinkResult) -> None:
    """Print a human-readable preview of changes."""
    print(f"\n{'='*60}")
    print("  AUTO-LINKER PREVIEW")
    print(f"{'='*60}\n")
    print(f"Pages scanned: {result.total_pages}")
    print(f"Pages to modify: {result.pages_modified}")
    print(f"Links to insert: {result.links_inserted}")
    print(f"Related sections to add: {result.related_sections_added}")

    if result.errors:
        print(f"\nErrors: {len(result.errors)}")
        for e in result.errors:
            print(f"  {e}")

    # Group insertions by file
    by_file: dict[str, list[LinkInsertion]] = {}
    for ins in result.insertions:
        by_file.setdefault(ins.file_path, []).append(ins)

    print("\n--- Changes by file ---\n")
    for file_path, insertions in sorted(by_file.items()):
        print(f"  {file_path}")
        for ins in insertions:
            target_short = ins.target_path.replace(".md", "")
            print(f"    L{ins.line_number}: \"{ins.anchor_text}\" -> [{target_short}]")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-link swarm-ai.org docs")
    parser.add_argument("--docs", default="docs", help="Path to docs/ directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    parser.add_argument("--apply", action="store_true", help="Actually write changes")
    parser.add_argument("--related", action="store_true", help="Add Related Pages sections")
    parser.add_argument("--max-links", type=int, default=5, help="Max new links per page")
    args = parser.parse_args()

    is_dry_run = not args.apply

    result = auto_link(
        docs_root=args.docs,
        dry_run=is_dry_run,
        add_related=args.related,
        max_links_per_page=args.max_links,
    )

    print_preview(result)

    if is_dry_run:
        print("  (dry run — use --apply to write changes)\n")
    else:
        print(f"  Applied {result.links_inserted} links across {result.pages_modified} files.\n")
