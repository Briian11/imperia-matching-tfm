from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from imperia_matching_tfm.property_extraction.rule_based import extract_property_from_text  # noqa: E402
from imperia_matching_tfm.property_extraction.pdf_text import extract_text_from_pdf  # noqa: E402
def main() -> int:
    args = parse_args()
    output_path = resolve_path(args.output)
    source_id = args.source_id or infer_source_id(args.input or args.pdf)
    try:
        text, source_mode = resolve_text(args)
    except FileNotFoundError as error:
        missing = error.filename or str(error)
        print(f"No se encontro el archivo: {missing}")
        print("Comprueba la ruta o guarda el PDF/texto dentro de data/inbox/.")
        return 2

    if not text.strip():
        print("No hay texto disponible para extraer la propiedad.")
        return 2

    try:
        property_payload = extract_property_from_text(
            text,
            source_id=source_id,
            source_url=None,
            title=args.title,
            source_mode=source_mode,
        )
    except ValueError as error:
        print(f"No se pudo extraer la propiedad: {error}")
        return 3

    existing = load_existing(output_path)
    upserted = [item for item in existing if item["id"] != property_payload["id"]]
    upserted.append(property_payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(upserted, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(property_payload, output_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Añade una propiedad al dataset desde PDF o texto local.")
    parser.add_argument("--pdf", default=None, help="PDF guardado desde el navegador con Cmd+P / imprimir.")
    parser.add_argument("--input", default=None, help="Archivo .txt de fallback o fuente manual.")
    parser.add_argument("--source-id", default=None, help="Identificador interno. Si se omite, se infiere.")
    parser.add_argument("--title", default=None, help="Titulo normalizado de la propiedad.")
    parser.add_argument(
        "--output",
        default="data/processed/user_properties.json",
        help="JSON de salida al que se añadira la propiedad.",
    )
    args = parser.parse_args()

    if not args.input and not args.pdf:
        parser.error("Debes indicar --pdf o --input.")
    return args


def resolve_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.pdf:
        return read_pdf_text(args.pdf), "pdf"

    if args.input:
        return resolve_path(args.input).read_text(encoding="utf-8"), "manual_text"

    return "", "unknown"


def read_pdf_text(value: str) -> str:
    pdf_path = resolve_path(value)
    try:
        text = extract_text_from_pdf(pdf_path)
    except RuntimeError as error:
        print(str(error))
        return ""
    if not text.strip():
        print("El PDF no contiene texto extraible. Prueba a guardar desde el navegador como PDF de pagina, no como captura.")
    return text


def infer_source_id(input_path: str | None) -> str:
    if input_path:
        stem = Path(input_path).stem
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
        return normalized[:80] or "property_manual"

    return "property_manual"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def print_summary(property_payload: dict, output_path: Path) -> None:
    print(f"Propiedad extraida: {property_payload['id']}")
    print(f"Titulo: {property_payload['title']}")
    print(f"Precio: {property_payload['price']}")
    print(f"Ciudad: {property_payload['city']}")
    print(f"Zona: {property_payload['zone']}")
    print(f"Habitaciones: {property_payload['bedrooms']}")
    print(f"Baños: {property_payload['bathrooms']}")
    print(f"Superficie construida: {property_payload['surface_m2']} m2")
    print(f"Superficie util: {property_payload['usable_surface_m2']} m2")
    print(f"Planta: {property_payload['floor']}")
    print(f"Año construccion: {property_payload['built_year']}")
    print(f"Features: {', '.join(property_payload['features'])}")
    print(f"Dataset actualizado: {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
