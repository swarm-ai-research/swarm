#!/usr/bin/env python3
"""
Publish research posts to Moltbook.

Reads a markdown post from research/posts/, strips frontmatter,
creates the post via the Moltbook API, solves the CAPTCHA verification
challenge, and tracks published IDs to prevent double-posting.

Usage:
    python -m swarm.scripts.publish_moltbook research/posts/circuit_breakers_dominate.md
    python -m swarm.scripts.publish_moltbook --dry-run research/posts/smarter_agents_earn_less.md
    python -m swarm.scripts.publish_moltbook --submolt aisafety research/posts/governance_lessons_70_runs.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = "https://www.moltbook.com/api/v1"
CREDENTIALS_PATH = Path.home() / ".config" / "moltbook" / "credentials.json"
PUBLISHED_PATH = Path("research/posts/.published.json")

# Number words for CAPTCHA solving
_ONES = [
    "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine",
]
_TEENS = [
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
]

# Regex patterns for number words with character-doubling tolerance.
# Built lazily on first use.
_NUMBER_PATTERNS: list[tuple[re.Pattern[str], float]] = []


def _word_to_fuzzy_regex(word: str) -> str:
    """Build a regex matching a word with arbitrary character repetition."""
    return "".join(f"{re.escape(c)}+" for c in word)


def _number_to_words(n: int) -> str:
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        t = _TENS[n // 10]
        o = _ONES[n % 10] if n % 10 else ""
        return f"{t} {o}".strip()
    if n < 1000:
        h = _ONES[n // 100]
        rem = n % 100
        if rem == 0:
            return f"{h} hundred"
        return f"{h} hundred {_number_to_words(rem)}"
    return str(n)


def _build_number_patterns() -> None:
    """Build regex patterns for numbers 2-999 (the CAPTCHA range)."""
    if _NUMBER_PATTERNS:
        return

    for n in range(2, 1000):
        words = _number_to_words(n)
        word_parts = words.split()
        regex_parts = [_word_to_fuzzy_regex(w) for w in word_parts]
        pattern_str = r"\s+".join(regex_parts)
        _NUMBER_PATTERNS.append(
            (re.compile(pattern_str, re.IGNORECASE), float(n))
        )

    # Sort descending so multi-word numbers match before their parts
    _NUMBER_PATTERNS.sort(key=lambda x: -x[1])


def _deobfuscate(text: str) -> str:
    """Remove injected punctuation and filler words."""
    # Remove known injected punctuation chars (from Moltbook ChallengeGenerator)
    cleaned = re.sub(r'[\^/~|\]}<*+\[\]\\]', '', text)
    # Remove dashes used as injected separators (between spaces or at word boundaries)
    cleaned = re.sub(r'(?<=\s)-(?=\s)', '', cleaned)
    cleaned = re.sub(r'(?<=[a-zA-Z])-(?=\s)', '', cleaned)
    # Also remove stray periods that appear in obfuscated text
    cleaned = re.sub(r'(?<=[a-zA-Z])\.(?=[a-zA-Z])', '', cleaned)
    # Remove filler words
    fillers = {"um", "uh", "erm", "like", "eh"}
    words = cleaned.split()
    words = [w for w in words if w.lower().strip(".,!?;:'\"") not in fillers]
    return " ".join(words)


def _find_numbers(text: str) -> list[float]:
    """Find all numbers in (possibly obfuscated) text."""
    _build_number_patterns()

    # Try digit numbers first
    digit_nums = [float(m.group()) for m in re.finditer(r'\b\d+(?:\.\d+)?\b', text)]
    if digit_nums:
        return digit_nums

    # Fuzzy-match spelled-out numbers
    numbers: list[float] = []
    remaining = text
    while remaining:
        best_match = None
        best_val = 0.0
        best_start = len(remaining)
        best_end = 0

        for pattern, val in _NUMBER_PATTERNS:
            m = pattern.search(remaining)
            if m and m.start() < best_start:
                best_match = m
                best_val = val
                best_start = m.start()
                best_end = m.end()

        if best_match is None:
            break

        numbers.append(best_val)
        remaining = remaining[best_end:]

    return numbers


def _detect_operation(text: str) -> str:
    """Detect math operation from challenge keywords."""
    # Collapse adjacent duplicate chars for robust keyword matching
    collapsed = re.sub(r'(.)\1+', r'\1', text.lower())

    if "per claw" in collapsed or "split" in collapsed or "divided" in collapsed:
        return "divide"
    if "remain" in collapsed or "lose" in collapsed or "left" in collapsed:
        return "subtract"
    # Explicit multiply keywords — check before addition keywords
    if "each" in collapsed or "multiply" in collapsed or "product" in collapsed:
        return "multiply"
    if any(kw in collapsed for kw in (
        "ads", "sum", "combined", "together", "plus", "total",
    )):
        return "add"
    if ("shel" in collapsed and "find" in collapsed) or (
        "how many" in collapsed and "more" in collapsed
    ):
        return "add"
    # force, distance, and default -> multiply
    return "multiply"


def _solve_captcha_llm(challenge_text: str) -> Optional[float]:
    """Solve CAPTCHA using an LLM.

    Tries in order: claude CLI, Anthropic API, OpenAI API.
    """
    prompt = (
        "Solve this obfuscated math CAPTCHA. The text has been mangled: "
        "letters may be doubled/tripled (e.g. 'ttwweennttyy' = 'twenty'), "
        "random punctuation injected (^/~|]}<*+), and filler words added "
        "(um, uh, like). Steps:\n"
        "1. Remove repeated letters to get the real words\n"
        "2. Identify the two numbers\n"
        "3. Identify the operation (add/subtract/multiply/divide)\n"
        "4. Compute the answer\n"
        "Return ONLY the final numerical answer (a single number).\n\n"
        f"Challenge: {challenge_text}"
    )

    # Try claude CLI first (works when running inside Claude Code)
    result = _solve_via_claude_cli(prompt)
    if result is not None:
        return result

    # Fall back to API calls
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    if os.environ.get("ANTHROPIC_API_KEY"):
        url = "https://api.anthropic.com/v1/messages"
        body = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = Request(url, data=body, method="POST")
        req.add_header("x-api-key", api_key)
        req.add_header("anthropic-version", "2023-06-01")
        req.add_header("Content-Type", "application/json")
    else:
        url = "https://api.openai.com/v1/chat/completions"
        body = json.dumps({
            "model": "gpt-4o-mini",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if os.environ.get("ANTHROPIC_API_KEY"):
                text = data["content"][0]["text"].strip()
            else:
                text = data["choices"][0]["message"]["content"].strip()
            m = re.search(r'-?\d+(?:\.\d+)?', text)
            if m:
                return round(float(m.group()), 2)
    except Exception as e:
        print(f"LLM API solver failed: {e}", file=sys.stderr)
    return None


def _solve_via_claude_cli(prompt: str) -> Optional[float]:
    """Solve using claude CLI (available inside Claude Code sessions)."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", "haiku"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            text = result.stdout.strip()
            m = re.search(r'-?\d+(?:\.\d+)?', text)
            if m:
                return round(float(m.group()), 2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _solve_captcha_regex(challenge_text: str) -> Optional[float]:
    """Solve CAPTCHA using regex-based number extraction."""
    cleaned = _deobfuscate(challenge_text)
    numbers = _find_numbers(cleaned)

    if len(numbers) < 2:
        return None

    a, b = numbers[0], numbers[1]
    op = _detect_operation(cleaned)

    if op == "add":
        result = a + b
    elif op == "subtract":
        result = a - b
    elif op == "divide":
        result = a / b if b != 0 else 0.0
    else:
        result = a * b

    return round(result, 2)


def solve_captcha(challenge_text: str) -> Optional[float]:
    """Solve a Moltbook obfuscated math CAPTCHA challenge.

    Uses LLM solver (claude CLI, Anthropic API, or OpenAI API) with
    regex-based cross-validation when both produce answers.
    """
    llm_result = _solve_captcha_llm(challenge_text)
    regex_result = _solve_captcha_regex(challenge_text)

    if llm_result is not None and regex_result is not None:
        if llm_result == regex_result:
            return llm_result
        # Disagreement: prefer regex for arithmetic, LLM for parsing
        print(f"  CAPTCHA solvers disagree: LLM={llm_result}, regex={regex_result}",
              file=sys.stderr)
        return regex_result

    return llm_result if llm_result is not None else regex_result


# ── API helpers ──────────────────────────────────────────────────────


def load_credentials(account: str = "current") -> dict:
    """Load Moltbook API credentials."""
    if not CREDENTIALS_PATH.exists():
        print(f"Error: No credentials at {CREDENTIALS_PATH}", file=sys.stderr)
        sys.exit(1)
    data: dict = json.loads(CREDENTIALS_PATH.read_text())
    if account not in data:
        print(f"Error: No '{account}' account in credentials", file=sys.stderr)
        print(f"Available: {', '.join(data.keys())}", file=sys.stderr)
        sys.exit(1)
    creds: dict = data[account]
    return creds


def load_published() -> dict:
    """Load the published posts tracking file."""
    if PUBLISHED_PATH.exists():
        result: dict = json.loads(PUBLISHED_PATH.read_text())
        return result
    return {}


def save_published(data: dict) -> None:
    """Save the published posts tracking file."""
    PUBLISHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLISHED_PATH.write_text(json.dumps(data, indent=2) + "\n")


def parse_post(filepath: Path) -> dict:
    """Parse a Moltbook post markdown file into title, content, submolt."""
    text = filepath.read_text()
    lines = text.split("\n")

    title = ""
    submolt = "general"
    content_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("**submolt:**"):
            raw = stripped.replace("**submolt:**", "").strip()
            submolt = raw.split("/")[-1] if "/" in raw else raw
        elif stripped.startswith("## ") and not title:
            title = stripped[3:].strip()
            content_start = i

    body_lines = lines[content_start:]
    content = "\n".join(body_lines).strip()

    return {"title": title, "content": content, "submolt": submolt}


def api_call(
    method: str, endpoint: str, api_key: str, data: Optional[dict] = None,
) -> dict:
    """Make an API call to Moltbook."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req, timeout=30) as resp:
            result: dict = json.loads(resp.read())
            return result
    except HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        try:
            err = json.loads(body_text)
        except (json.JSONDecodeError, ValueError):
            err = {"error": body_text}

        if e.code == 429:
            retry = err.get("retry_after", "unknown")
            print(f"Rate limited. Retry after {retry} seconds.", file=sys.stderr)
            sys.exit(1)

        print(f"API error {e.code}: {err}", file=sys.stderr)
        sys.exit(1)


def create_submolt(
    api_key: str, name: str, display_name: str, description: str,
) -> dict:
    """Create a new submolt on Moltbook."""
    return api_call("POST", "/submolts", api_key, {
        "name": name,
        "display_name": display_name,
        "description": description,
    })


# ── Main publish flow ────────────────────────────────────────────────


def publish_post(
    filepath: Path,
    submolt_override: Optional[str] = None,
    dry_run: bool = False,
    account: str = "current",
) -> Optional[str]:
    """Publish a post to Moltbook. Returns the post ID on success."""
    published = load_published()
    file_key = str(filepath)

    if file_key in published:
        print(f"Already published: {filepath.name}")
        print(f"  Post ID: {published[file_key]['post_id']}")
        print(f"  URL: {published[file_key].get('url', 'unknown')}")
        return str(published[file_key]["post_id"])

    post = parse_post(filepath)
    if submolt_override:
        post["submolt"] = submolt_override

    print(f"Title:   {post['title']}")
    print(f"Submolt: {post['submolt']}")
    print(f"Length:  {len(post['content'])} chars")
    print()

    if dry_run:
        print("--- DRY RUN (content preview) ---")
        preview = post["content"][:500]
        print(preview)
        if len(post["content"]) > 500:
            print(f"... ({len(post['content']) - 500} more chars)")
        return None

    creds = load_credentials(account)

    # Step 1: Create the post
    print("Creating post...")
    result = api_call("POST", "/posts", creds["api_key"], {
        "title": post["title"],
        "content": post["content"],
        "submolt": post["submolt"],
    })

    if not result.get("success"):
        print(f"Failed to create post: {result}", file=sys.stderr)
        sys.exit(1)

    post_id: str = result["post"]["id"]
    post_url = result["post"].get("url", "")
    print(f"Post created: {post_id}")

    # Step 2: Solve CAPTCHA verification
    verification = result.get("verification", {})
    if verification:
        challenge = verification.get("challenge", "")
        verify_code = verification.get("code", "")
        print(f"Challenge: {challenge}")
        print("Solving verification challenge...")

        answer = solve_captcha(challenge)
        if answer is None:
            print(f"Could not solve CAPTCHA: {challenge}", file=sys.stderr)
            print("Post created but not verified. Verify manually.", file=sys.stderr)
            return post_id

        print(f"Answer: {answer:.2f}")
        url = f"{BASE_URL}/verify"
        body = json.dumps({
            "verification_code": verify_code,
            "answer": f"{answer:.2f}",
        }).encode()
        req = Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {creds['api_key']}")
        req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req, timeout=30) as resp:
                verify_result: dict = json.loads(resp.read())
        except HTTPError as e:
            err_body = e.read().decode() if e.fp else ""
            print(f"Verification error {e.code}: {err_body}", file=sys.stderr)
            verify_result = {"success": False}

        if verify_result.get("success"):
            print("Verified and published!")
        else:
            print("Post created but not verified.", file=sys.stderr)
    else:
        print("No verification required.")

    # Step 3: Track publication
    published[file_key] = {
        "post_id": post_id,
        "url": f"https://www.moltbook.com{post_url}",
        "submolt": post["submolt"],
        "title": post["title"],
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    save_published(published)
    print(f"Tracked in {PUBLISHED_PATH}")

    return post_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish research posts to Moltbook",
    )
    parser.add_argument("file", type=Path, help="Markdown post file to publish")
    parser.add_argument(
        "--submolt", help="Override the submolt from the post frontmatter",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be posted",
    )
    parser.add_argument(
        "--create-submolt",
        metavar="NAME",
        help="Create a submolt before posting (e.g. multiagent-safety)",
    )
    parser.add_argument(
        "--account",
        default="current",
        help="Credential profile to use (default: current)",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    if args.create_submolt:
        creds = load_credentials(args.account)
        print(f"Creating submolt: {args.create_submolt}")
        result = create_submolt(
            creds["api_key"],
            name=args.create_submolt,
            display_name=args.create_submolt.replace("-", " ").title(),
            description="Distributional safety research for multi-agent AI systems.",
        )
        print(f"Result: {result}")
        print()

    publish_post(
        args.file,
        submolt_override=args.submolt,
        dry_run=args.dry_run,
        account=args.account,
    )


if __name__ == "__main__":
    main()
