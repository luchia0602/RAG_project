
"""Extract and clean text from a PDF while preserving page boundaries."""

import re
from collections import Counter
from pypdf import PdfReader


def clean_line(line):
    """Normalize whitespace without changing the words in a rule."""
    return re.sub(r"\s+", " ", line).strip()


def read_pdf(file):
    """Extract text page by page and remove common layout artifacts."""

    reader = PdfReader(file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Normalize line endings and remove null characters.
        text = text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        lines = [clean_line(line) for line in text.split("\n")]
        lines = [line for line in lines if line]

        pages.append(lines)

        print(
            f"Extracted page {page_number}/{len(reader.pages)}: "
            f"{len(lines)} non-empty lines"
        )

    # Identify short lines repeated across many pages.
    # These are often running headers or footers.
    line_counts = Counter()

    for lines in pages:
        # Count a line at most once per page.
        line_counts.update(set(
            line for line in lines if len(line) <= 100
        ))

    min_repeated_pages = max(3, int(len(pages) * 0.20))

    repeated_lines = {
        line
        for line, count in line_counts.items()
        if count >= min_repeated_pages
        and (
            "CORE RULES" in line.upper()
            or re.fullmatch(r"\d{1,4}", line)
            or re.fullmatch(r"PAGE\s+\d+", line.upper())
        )
    }

    cleaned_pages = []

    for lines in pages:
        cleaned_lines = []

        for line in lines:
            if line in repeated_lines:
                continue

            cleaned_lines.append(line)

        page_text = "\n".join(cleaned_lines).strip()
        cleaned_pages.append(page_text)

    # Form-feed marks the boundary between PDF pages.
    # The chunker will use this to avoid joining page fragments
    # without a separator.
    return "\n\f\n".join(cleaned_pages)
