"""Export SWARM papers to PDF for AgentRxiv submission.

This module provides utilities for converting LaTeX papers
to PDF format suitable for AgentRxiv upload.
"""

import base64
import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from swarm.research.platforms import Paper

logger = logging.getLogger(__name__)


class PDFExportError(Exception):
    """Error during PDF export."""

    pass


def check_pdflatex() -> bool:
    """Check if pdflatex is available.

    Returns:
        True if pdflatex is installed and accessible.
    """
    try:
        result = subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def paper_to_pdf(
    paper: Paper,
    output_path: str | Path | None = None,
    keep_temp: bool = False,
) -> Path:
    """Convert a LaTeX paper to PDF.

    Args:
        paper: Paper with LaTeX source.
        output_path: Optional output path for the PDF.
            If not provided, a temporary file is created.
        keep_temp: Whether to keep temporary files on error.

    Returns:
        Path to the generated PDF file.

    Raises:
        PDFExportError: If PDF generation fails.
        ValueError: If paper has no LaTeX source.
    """
    if not paper.source:
        raise ValueError("Paper has no LaTeX source")

    if not check_pdflatex():
        raise PDFExportError(
            "pdflatex not found. Install with: sudo apt install texlive-latex-base"
        )

    with tempfile.TemporaryDirectory() as tmpdir_str:
        work_dir = Path(tmpdir_str)

        try:
            # Write main LaTeX file
            tex_path = work_dir / "paper.tex"
            tex_path.write_text(paper.source)

            # Write bibliography if present
            if paper.bib:
                bib_path = work_dir / "references.bib"
                bib_path.write_text(paper.bib)

            # Write any images
            for name, data in paper.images.items():
                img_path = work_dir / name
                img_path.write_bytes(base64.b64decode(data))

            # First pdflatex pass
            _run_pdflatex(work_dir, "paper.tex")

            # Run bibtex if bibliography exists
            if paper.bib:
                _run_bibtex(work_dir, "paper")
                # Additional pdflatex pass after bibtex
                _run_pdflatex(work_dir, "paper.tex")

            # Final pdflatex pass
            _run_pdflatex(work_dir, "paper.tex")

            # Check for output
            pdf_path = work_dir / "paper.pdf"
            if not pdf_path.exists():
                # Try to get error from log
                log_path = work_dir / "paper.log"
                if log_path.exists():
                    log_content = log_path.read_text()
                    # Find error lines
                    errors = [
                        line for line in log_content.split("\n")
                        if line.startswith("!")
                    ]
                    if errors:
                        raise PDFExportError(
                            f"LaTeX compilation failed: {errors[0]}"
                        )
                raise PDFExportError("PDF compilation failed - no output file")

            # Copy to final destination
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(pdf_path, output_path)
                return output_path
            else:
                # Copy to persistent temp location
                final_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.pdf"
                shutil.copy(pdf_path, final_path)
                return final_path

        except Exception:
            if keep_temp:
                # Copy temp dir for debugging
                debug_dir = Path(f"./pdf_export_debug_{uuid.uuid4().hex[:8]}")
                shutil.copytree(work_dir, debug_dir)
                logger.error(f"Debug files saved to {debug_dir}")
            raise


def _run_pdflatex(
    working_dir: Path,
    tex_file: str,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """Run pdflatex on a file.

    Args:
        working_dir: Directory containing the tex file.
        tex_file: Name of the tex file.
        timeout: Maximum seconds to wait.

    Returns:
        CompletedProcess result.
    """
    result = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_file,
        ],
        cwd=working_dir,
        capture_output=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        logger.debug(f"pdflatex output: {result.stdout.decode()}")

    return result


def _run_bibtex(
    working_dir: Path,
    aux_name: str,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Run bibtex on an aux file.

    Args:
        working_dir: Directory containing the aux file.
        aux_name: Name of the aux file (without extension).
        timeout: Maximum seconds to wait.

    Returns:
        CompletedProcess result.
    """
    result = subprocess.run(
        ["bibtex", aux_name],
        cwd=working_dir,
        capture_output=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        logger.debug(f"bibtex output: {result.stdout.decode()}")

    return result


def markdown_to_pdf(
    content: str,
    title: str = "Paper",
    output_path: str | Path | None = None,
) -> Path:
    """Convert Markdown content to PDF.

    This is a simpler alternative for papers that don't need
    full LaTeX features.

    Args:
        content: Markdown content.
        title: Paper title.
        output_path: Optional output path.

    Returns:
        Path to the generated PDF.

    Raises:
        PDFExportError: If conversion fails.
    """
    # Check for pandoc
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        raise PDFExportError(
            "pandoc not found. Install with: sudo apt install pandoc"
        ) from None

    with tempfile.TemporaryDirectory() as tmpdir_str:
        work_dir = Path(tmpdir_str)

        # Write markdown
        md_path = work_dir / "paper.md"
        md_path.write_text(f"# {title}\n\n{content}")

        # Convert with pandoc
        pdf_path = work_dir / "paper.pdf"

        result = subprocess.run(
            [
                "pandoc",
                str(md_path),
                "-o", str(pdf_path),
                "--pdf-engine=pdflatex",
            ],
            capture_output=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise PDFExportError(
                f"pandoc failed: {result.stderr.decode()}"
            )

        if not pdf_path.exists():
            raise PDFExportError("PDF not generated")

        # Copy to final destination
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(pdf_path, output_path)
            return output_path
        else:
            final_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.pdf"
            shutil.copy(pdf_path, final_path)
            return final_path


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text content.

    Raises:
        PDFExportError: If extraction fails.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            raise PDFExportError(
                "pypdf or PyPDF2 required. Install with: pip install pypdf"
            ) from None

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise PDFExportError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    text_parts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)

    return "\n\n".join(text_parts)
