"""
Meta description generator for swarm-ai.org docs.

Generates concise (<160 char) meta descriptions from page content
and inserts them as YAML frontmatter `description:` fields.

No API keys required.

Usage:
    python scripts/seo/meta_generator.py --docs docs --dry-run     # preview
    python scripts/seo/meta_generator.py --docs docs --apply        # write
    python scripts/seo/meta_generator.py --docs docs --apply --overwrite  # replace existing
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Description extraction
# ---------------------------------------------------------------------------

# Sentences to skip (common but uninformative openers)
_SKIP_PATTERNS = [
    re.compile(r"^(this|the following|here|below|note|warning|todo|fixme)", re.IGNORECASE),
    re.compile(r"^#{1,6}\s"),  # headings
    re.compile(r"^\|"),  # table rows
    re.compile(r"^```"),  # code blocks
    re.compile(r"^<"),  # HTML tags
    re.compile(r"^!\["),  # images
    re.compile(r"^---"),  # horizontal rules / frontmatter
    re.compile(r"^\*\*?(Disclaimer|Note|Warning)", re.IGNORECASE),
    re.compile(r"^\s*$"),  # blank lines
]


def _extract_description(text: str, max_len: int = 155) -> str:
    """
    Extract a meta description from Markdown content.

    Strategy:
    1. If the page has an explicit subtitle/tagline (first non-heading paragraph), use it.
    2. Otherwise, use the first meaningful sentence.
    3. Truncate to max_len at a word boundary, append ellipsis if needed.
    """
    # Strip frontmatter
    body = text
    if body.startswith("---"):
        end = body.find("---", 3)
        if end != -1:
            body = body[end + 3:].strip()

    # Strip HTML blocks and style tags
    body = re.sub(r"<style[\s\S]*?</style>", "", body)
    body = re.sub(r"<[^>]+>", "", body)

    # Strip code blocks
    body = re.sub(r"```[\s\S]*?```", "", body)

    # Strip inline formatting but keep text
    body = re.sub(r"\*\*([^*]+)\*\*", r"\1", body)
    body = re.sub(r"\*([^*]+)\*", r"\1", body)
    body = re.sub(r"`([^`]+)`", r"\1", body)

    # Strip markdown links, keep text
    body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)

    # Strip admonition markers
    body = re.sub(r"^!!!\s+\w+.*$", "", body, flags=re.MULTILINE)

    lines = body.split("\n")

    # Find the first meaningful paragraph
    candidates: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        # Skip non-content lines
        if any(p.match(line) for p in _SKIP_PATTERNS):
            continue

        # Skip headings
        if line.startswith("#"):
            continue

        # Found a content line — collect the full paragraph
        para_lines = [line]
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("#"):
            next_line = lines[i].strip()
            if any(p.match(next_line) for p in _SKIP_PATTERNS):
                break
            para_lines.append(next_line)
            i += 1

        paragraph = " ".join(para_lines)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if len(paragraph) > 30:  # minimum useful length
            candidates.append(paragraph)
            if len(candidates) >= 2:
                break

    if not candidates:
        return ""

    # Prefer the first candidate (usually the intro paragraph)
    desc = candidates[0]

    # Clean up any remaining markdown artifacts
    desc = re.sub(r"\$\$[^$]*\$\$", "", desc)
    desc = re.sub(r"\$[^$]+\$", "", desc)
    desc = desc.strip()

    if not desc:
        return ""

    # Truncate at word boundary
    if len(desc) > max_len:
        truncated = desc[:max_len]
        # Cut at last space
        last_space = truncated.rfind(" ")
        if last_space > max_len * 0.6:
            truncated = truncated[:last_space]
        desc = truncated.rstrip(".,;:- ") + "..."

    return desc


# ---------------------------------------------------------------------------
# Frontmatter manipulation
# ---------------------------------------------------------------------------

def _has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") or text.startswith("---\r\n")


def _get_existing_description(text: str) -> str | None:
    """Extract existing description from frontmatter, if any."""
    if not _has_frontmatter(text):
        return None
    end = text.find("---", 3)
    if end == -1:
        return None
    fm = text[3:end]
    m = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE)
    if m:
        return m.group(1).strip().strip('"').strip("'")
    return None


def _insert_description(text: str, description: str) -> str:
    """Insert or update description in frontmatter."""
    # Escape quotes in description
    safe_desc = description.replace('"', '\\"')

    if _has_frontmatter(text):
        end = text.find("---", 3)
        if end == -1:
            return text
        fm = text[3:end]
        body_after = text[end:]

        # Check if description already exists
        if re.search(r"^description:", fm, re.MULTILINE):
            # Replace it
            fm = re.sub(
                r"^description:.*$",
                f'description: "{safe_desc}"',
                fm,
                flags=re.MULTILINE,
            )
        else:
            # Add it as first field after opening ---
            fm = f'description: "{safe_desc}"\n' + fm

        return "---\n" + fm + body_after
    else:
        # Create new frontmatter
        return f'---\ndescription: "{safe_desc}"\n---\n\n' + text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@dataclass
class MetaResult:
    """Result of the meta description generation pass."""
    total_pages: int
    descriptions_generated: int
    descriptions_skipped: int  # already had one
    pages_modified: int
    changes: list[dict]  # {path, description, action: "add"|"update"|"skip"}
    errors: list[str]


def generate_meta_descriptions(
    docs_root: str | Path = "docs",
    dry_run: bool = True,
    overwrite: bool = False,
    max_len: int = 155,
    skip_patterns: list[str] | None = None,
) -> MetaResult:
    """
    Generate and insert meta descriptions for all docs.

    Args:
        docs_root: Path to docs/ directory
        dry_run: Preview only
        overwrite: Replace existing descriptions
        max_len: Max description length
        skip_patterns: File patterns to skip
    """
    root = Path(docs_root)
    md_files = sorted(root.rglob("*.md"))

    skip = set(skip_patterns or [])

    changes: list[dict] = []
    modified = 0
    skipped = 0
    generated = 0
    errors: list[str] = []

    for file_path in md_files:
        rel = str(file_path.relative_to(root))

        if rel in skip or any(rel.endswith(s) for s in skip):
            continue

        try:
            text = file_path.read_text(errors="replace")
        except Exception as e:
            errors.append(f"Read error {rel}: {e}")
            continue

        existing = _get_existing_description(text)

        if existing and not overwrite:
            changes.append({"path": rel, "description": existing, "action": "skip"})
            skipped += 1
            continue

        description = _extract_description(text, max_len=max_len)
        if not description:
            changes.append({"path": rel, "description": "", "action": "skip_empty"})
            skipped += 1
            continue

        action = "update" if existing else "add"
        changes.append({"path": rel, "description": description, "action": action})
        generated += 1

        new_text = _insert_description(text, description)
        if new_text != text:
            modified += 1
            if not dry_run:
                file_path.write_text(new_text)

    return MetaResult(
        total_pages=len(md_files),
        descriptions_generated=generated,
        descriptions_skipped=skipped,
        pages_modified=modified,
        changes=changes,
        errors=errors,
    )


def print_preview(result: MetaResult) -> None:
    """Print a human-readable preview."""
    print(f"\n{'='*60}")
    print("  META DESCRIPTION GENERATOR PREVIEW")
    print(f"{'='*60}\n")
    print(f"Pages scanned: {result.total_pages}")
    print(f"Descriptions to generate: {result.descriptions_generated}")
    print(f"Pages skipped: {result.descriptions_skipped}")
    print(f"Pages to modify: {result.pages_modified}")

    if result.errors:
        print(f"\nErrors: {len(result.errors)}")
        for e in result.errors:
            print(f"  {e}")

    # Show generated descriptions
    adds = [c for c in result.changes if c["action"] in ("add", "update")]
    if adds:
        print("\n--- Generated Descriptions ---\n")
        for c in adds[:30]:
            action_label = "+" if c["action"] == "add" else "~"
            desc = c["description"]
            # Truncate display
            if len(desc) > 80:
                desc = desc[:77] + "..."
            print(f"  [{action_label}] {c['path']}")
            print(f"      {desc}")
            print()
        if len(adds) > 30:
            print(f"  ... and {len(adds) - 30} more\n")

    # Show pages that couldn't generate a description
    empties = [c for c in result.changes if c["action"] == "skip_empty"]
    if empties:
        print(f"\n--- Could not generate ({len(empties)} pages) ---")
        for c in empties[:10]:
            print(f"  {c['path']}")
        if len(empties) > 10:
            print(f"  ... and {len(empties) - 10} more")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate meta descriptions for swarm-ai.org docs")
    parser.add_argument("--docs", default="docs", help="Path to docs/ directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    parser.add_argument("--apply", action="store_true", help="Actually write changes")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing descriptions")
    parser.add_argument("--max-len", type=int, default=155, help="Max description length")
    args = parser.parse_args()

    is_dry_run = not args.apply

    result = generate_meta_descriptions(
        docs_root=args.docs,
        dry_run=is_dry_run,
        overwrite=args.overwrite,
        max_len=args.max_len,
    )

    print_preview(result)

    if is_dry_run:
        print("  (dry run — use --apply to write changes)\n")
    else:
        print(f"  Applied {result.descriptions_generated} descriptions across {result.pages_modified} files.\n")
