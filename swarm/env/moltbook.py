"""Moltbook domain model with anti-human CAPTCHA challenges."""

import random
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ObfuscationStyle(Enum):
    """Obfuscation styles for challenges."""

    ALTERNATING_CASE = "alternating_case"
    INJECTED_PUNCTUATION = "injected_punctuation"
    SPELLED_NUMBERS = "spelled_numbers"
    FILLER_WORDS = "filler_words"


class MathOperation(Enum):
    """Math operations used in challenges."""

    ADD = "add"
    MULTIPLY = "multiply"
    SUBTRACT = "subtract"
    DIVIDE = "divide"


@dataclass
class MathChallenge:
    """A single obfuscated math challenge."""

    challenge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str = ""
    clean_text: str = ""
    answer: float = 0.0
    operation: MathOperation = MathOperation.ADD
    operands: List[float] = field(default_factory=list)
    expires_at_step: int = 0
    obfuscation_level: float = 0.0


class ChallengeGenerator:
    """Generate obfuscated lobster-themed math challenges."""

    _PUNCTUATION = ["^", "/", "~", "|", "]", "}", "<", "*", "+"]
    _FILLER_WORDS = ["um", "uh", "erm", "like", "eh"]

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._templates: List[Tuple[str, MathOperation]] = [
            (
                "A lobster exerts {a} newtons per claw and has {b} claws. "
                "How much total force?",
                MathOperation.MULTIPLY,
            ),
            (
                "A lobster swims at {a} meters per second for {b} seconds. "
                "How far does it go?",
                MathOperation.MULTIPLY,
            ),
            (
                "A lobster has {a} shells and finds {b} more. How many shells total?",
                MathOperation.ADD,
            ),
            (
                "A lobster starts with {a} newtons of force and loses {b}. "
                "How much remains?",
                MathOperation.SUBTRACT,
            ),
            (
                "A lobster splits {a} newtons across {b} claws. How much per claw?",
                MathOperation.DIVIDE,
            ),
        ]

    def generate(self, difficulty: float = 0.5) -> MathChallenge:
        """Generate a math challenge with obfuscation."""
        difficulty = max(0.0, min(1.0, difficulty))
        template, operation = self._rng.choice(self._templates)

        min_val = 2
        max_val = int(10 + 90 * difficulty)
        if operation == MathOperation.DIVIDE:
            b = self._rng.randint(min_val, max_val)
            quotient = self._rng.randint(min_val, max_val)
            a = b * quotient
        else:
            a = self._rng.randint(min_val, max_val)
            b = self._rng.randint(min_val, max_val)

        clean_text = template.format(a=a, b=b)
        answer = self._solve(operation, float(a), float(b))
        raw_text = self._obfuscate(clean_text, difficulty)

        return MathChallenge(
            raw_text=raw_text,
            clean_text=clean_text,
            answer=answer,
            operation=operation,
            operands=[float(a), float(b)],
            obfuscation_level=difficulty,
        )

    @staticmethod
    def _solve(operation: MathOperation, a: float, b: float) -> float:
        if operation == MathOperation.ADD:
            result = a + b
        elif operation == MathOperation.SUBTRACT:
            result = a - b
        elif operation == MathOperation.MULTIPLY:
            result = a * b
        else:
            result = a / b if b != 0 else 0.0
        return round(result, 2)

    def _obfuscate(self, text: str, level: float) -> str:
        styles = [
            ObfuscationStyle.SPELLED_NUMBERS,
            ObfuscationStyle.INJECTED_PUNCTUATION,
            ObfuscationStyle.FILLER_WORDS,
            ObfuscationStyle.ALTERNATING_CASE,
        ]
        layers = 1 + int(level * (len(styles) - 1))
        chosen = styles[:layers]

        result = text
        if ObfuscationStyle.SPELLED_NUMBERS in chosen:
            result = self._spell_numbers(result)
        if ObfuscationStyle.INJECTED_PUNCTUATION in chosen:
            density = 0.08 + 0.25 * level
            result = self._inject_punctuation(result, density)
        if ObfuscationStyle.FILLER_WORDS in chosen:
            rate = 0.05 + 0.2 * level
            result = self._insert_filler(result, rate)
        if ObfuscationStyle.ALTERNATING_CASE in chosen:
            result = self._alternate_case(result)

        return result

    def _spell_numbers(self, text: str) -> str:
        def _replace(match: re.Match) -> str:
            number = float(match.group(0))
            return self._number_to_garbled_word(number)

        return re.sub(r"\b\d+(?:\.\d+)?\b", _replace, text)

    def _number_to_garbled_word(self, n: float) -> str:
        if n.is_integer():
            words = self._number_to_words(int(n))
        else:
            integer = int(n)
            fractional = str(n).split(".", maxsplit=1)[1]
            words = self._number_to_words(integer) + " point " + " ".join(fractional)
        return self._alternate_case(words)

    @staticmethod
    def _number_to_words(n: int) -> str:
        ones = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        teens = [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        tens = [
            "",
            "",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]
        if n < 10:
            return ones[n]
        if n < 20:
            return teens[n - 10]
        if n < 100:
            tens_word = tens[n // 10]
            ones_word = ones[n % 10] if n % 10 != 0 else ""
            return f"{tens_word} {ones_word}".strip()
        if n < 1000:
            remainder = n % 100
            if remainder == 0:
                return f"{ones[n // 100]} hundred"
            return f"{ones[n // 100]} hundred {ChallengeGenerator._number_to_words(remainder)}"
        return str(n)

    def _inject_punctuation(self, text: str, density: float) -> str:
        out: List[str] = []
        for ch in text:
            out.append(ch)
            if ch.isalpha() and self._rng.random() < density:
                out.append(self._rng.choice(self._PUNCTUATION))
        return "".join(out)

    @staticmethod
    def _alternate_case(text: str) -> str:
        out: List[str] = []
        upper = True
        for ch in text:
            if ch.isalpha():
                out.append(ch.upper() if upper else ch.lower())
                upper = not upper
            else:
                out.append(ch)
        return "".join(out)

    def _insert_filler(self, text: str, rate: float) -> str:
        words = text.split()
        out: List[str] = []
        for word in words:
            out.append(word)
            if self._rng.random() < rate:
                out.append(self._rng.choice(self._FILLER_WORDS))
        return " ".join(out)


class ContentStatus(Enum):
    """Lifecycle status for Moltbook content."""

    PENDING_VERIFICATION = "pending_verification"
    PUBLISHED = "published"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass
class MoltbookPost:
    """Post or comment in the Moltbook feed."""

    post_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    author_id: str = ""
    content: str = ""
    submolt: str = "general"
    status: ContentStatus = ContentStatus.PENDING_VERIFICATION
    challenge: Optional[MathChallenge] = None
    created_at_step: int = 0
    published_at_step: Optional[int] = None
    upvotes: int = 0
    downvotes: int = 0
    parent_id: Optional[str] = None

    @property
    def net_votes(self) -> int:
        return self.upvotes - self.downvotes

    def to_dict(self, include_challenge: bool = False) -> Dict:
        data: Dict[str, object] = {
            "post_id": self.post_id,
            "author_id": self.author_id,
            "content": self.content,
            "submolt": self.submolt,
            "status": self.status.value,
            "created_at_step": self.created_at_step,
            "published_at_step": self.published_at_step,
            "upvotes": self.upvotes,
            "downvotes": self.downvotes,
            "parent_id": self.parent_id,
        }
        if include_challenge and self.challenge is not None:
            data["challenge"] = {
                "challenge_id": self.challenge.challenge_id,
                "raw_text": self.challenge.raw_text,
                "clean_text": self.challenge.clean_text,
                "answer": self.challenge.answer,
                "operation": self.challenge.operation.value,
                "expires_at_step": self.challenge.expires_at_step,
                "obfuscation_level": self.challenge.obfuscation_level,
            }
        return data


class MoltbookFeed:
    """Feed with verification gate for Moltbook."""

    def __init__(self, max_content_length: int = 10000) -> None:
        self.max_content_length = max_content_length
        self._posts: Dict[str, MoltbookPost] = {}
        self._posts_by_author: Dict[str, List[str]] = {}
        self._replies_by_parent: Dict[str, List[str]] = {}

    def submit_content(
        self,
        author_id: str,
        content: str,
        submolt: str,
        current_step: int,
        challenge: Optional[MathChallenge] = None,
        parent_id: Optional[str] = None,
    ) -> MoltbookPost:
        if len(content) > self.max_content_length:
            raise ValueError(
                f"Content length {len(content)} exceeds maximum {self.max_content_length}"
            )
        if parent_id and parent_id not in self._posts:
            raise ValueError(f"Parent post {parent_id} does not exist")

        post = MoltbookPost(
            author_id=author_id,
            content=content,
            submolt=submolt,
            status=ContentStatus.PENDING_VERIFICATION,
            challenge=challenge,
            created_at_step=current_step,
            parent_id=parent_id,
        )
        self._posts[post.post_id] = post
        self._posts_by_author.setdefault(author_id, []).append(post.post_id)
        if parent_id:
            self._replies_by_parent.setdefault(parent_id, []).append(post.post_id)
        return post

    def get_post(self, post_id: str) -> Optional[MoltbookPost]:
        return self._posts.get(post_id)

    def get_pending_for_agent(self, agent_id: str) -> List[MoltbookPost]:
        post_ids = self._posts_by_author.get(agent_id, [])
        return [
            self._posts[pid]
            for pid in post_ids
            if self._posts[pid].status == ContentStatus.PENDING_VERIFICATION
        ]

    def get_published_posts(
        self,
        submolt: Optional[str] = None,
        limit: int = 50,
        sort: str = "top",
    ) -> List[MoltbookPost]:
        posts = [
            p
            for p in self._posts.values()
            if p.status == ContentStatus.PUBLISHED
            and (submolt is None or p.submolt == submolt)
        ]
        if sort == "new":
            posts.sort(
                key=lambda p: p.published_at_step or p.created_at_step, reverse=True
            )
        elif sort == "old":
            posts.sort(key=lambda p: p.published_at_step or p.created_at_step)
        else:
            posts.sort(
                key=lambda p: (p.net_votes, p.published_at_step or 0), reverse=True
            )
        return posts[:limit]

    def verify_content(self, post_id: str, answer: float, current_step: int) -> bool:
        post = self._posts.get(post_id)
        if not post or post.status != ContentStatus.PENDING_VERIFICATION:
            return False
        if post.challenge is None:
            return False
        if current_step > post.challenge.expires_at_step:
            post.status = ContentStatus.EXPIRED
            return False

        expected = round(float(post.challenge.answer), 2)
        provided = round(float(answer), 2)
        if provided != expected:
            post.status = ContentStatus.REJECTED
            return False

        post.status = ContentStatus.PUBLISHED
        post.published_at_step = current_step
        return True

    def expire_unverified(self, current_step: int) -> List[str]:
        expired: List[str] = []
        for post in self._posts.values():
            if post.status != ContentStatus.PENDING_VERIFICATION:
                continue
            if post.challenge is None:
                continue
            if current_step > post.challenge.expires_at_step:
                post.status = ContentStatus.EXPIRED
                expired.append(post.post_id)
        return expired

    def vote(self, post_id: str, direction: int) -> bool:
        post = self._posts.get(post_id)
        if not post or post.status != ContentStatus.PUBLISHED:
            return False
        if direction > 0:
            post.upvotes += 1
        elif direction < 0:
            post.downvotes += 1
        else:
            return False
        return True

    def get_replies(self, post_id: str) -> List[MoltbookPost]:
        reply_ids = self._replies_by_parent.get(post_id, [])
        return [self._posts[rid] for rid in reply_ids if rid in self._posts]

    def clear(self) -> None:
        self._posts.clear()
        self._posts_by_author.clear()
        self._replies_by_parent.clear()
