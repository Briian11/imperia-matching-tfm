from __future__ import annotations

from pathlib import Path


def extract_text_from_pdf(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(str(path))

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Para leer PDFs instala la dependencia opcional: python3 -m pip install pypdf"
        ) from error

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(text for text in pages if text.strip())
