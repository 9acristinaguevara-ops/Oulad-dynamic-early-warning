"""Intenta descargar y extraer OULAD.

Uso:
    python scripts\\01_download_oulad.py
    python scripts\\01_download_oulad.py --local-zip data/raw/downloads/oulad.zip

La web oficial puede cambiar. El script inspecciona enlaces .zip de la página y
prueba candidatos; si falla, muestra instrucciones de descarga manual.
"""
from pathlib import Path
import argparse
import re
import sys
import zipfile
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "data" / "raw" / "downloads"
CSV_DIR = ROOT / "data" / "raw" / "csv"
OFFICIAL_PAGE = "https://analyse.kmi.open.ac.uk/open_dataset"
EXPECTED = {
    "assessments.csv", "courses.csv", "studentAssessment.csv",
    "studentInfo.csv", "studentRegistration.csv", "studentVle.csv", "vle.csv"
}
HEADERS = {"User-Agent": "OULAD-research-project/0.1 (academic data download)"}


def extract_zip(zip_path: Path) -> bool:
    """Extrae solo los CSV esperados y evita rutas arbitrarias del ZIP."""
    print(f"Revisando archivo comprimido: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = []
            for name in archive.namelist():
                if Path(name).name in EXPECTED and not name.endswith("/"):
                    members.append(name)
            found = {Path(name).name for name in members}
            missing = EXPECTED - found
            if missing:
                print("El ZIP no contiene todos los CSV esperados.")
                print("Faltan:", ", ".join(sorted(missing)))
                return False
            CSV_DIR.mkdir(parents=True, exist_ok=True)
            for member in members:
                target_name = Path(member).name
                target = CSV_DIR / target_name
                with archive.open(member) as source, target.open("wb") as dest:
                    dest.write(source.read())
                print(f"Extraído: {target_name}")
        print(f"\nExtracción completada en: {CSV_DIR}")
        return True
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"No se pudo leer el ZIP: {exc}")
        return False


def find_zip_links(page_url: str) -> list[str]:
    """Busca enlaces que parezcan archivos ZIP en la página oficial."""
    response = requests.get(page_url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = urljoin(page_url, tag["href"])
        if ".zip" in urlparse(href).path.lower():
            if href not in links:
                links.append(href)
    return links


def download_candidate(url: str, destination: Path) -> bool:
    """Descarga a un archivo temporal y lo conserva si parece un ZIP."""
    print(f"Intentando descargar: {url}")
    temp = destination.with_suffix(".part")
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            with temp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if not zipfile.is_zipfile(temp):
            print("El enlace no devolvió un ZIP válido; se descarta.")
            temp.unlink(missing_ok=True)
            return False
        temp.replace(destination)
        print(f"Descarga guardada en: {destination}")
        return True
    except requests.RequestException as exc:
        print(f"Falló la descarga: {exc}")
        temp.unlink(missing_ok=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-zip", type=str, help="Ruta al ZIP de OULAD descargado manualmente")
    args = parser.parse_args()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if args.local_zip:
        local_zip = Path(args.local_zip)
        if not local_zip.is_absolute():
            local_zip = (Path.cwd() / local_zip).resolve()
        if not local_zip.exists():
            print(f"No existe el archivo indicado: {local_zip}")
            return 2
        return 0 if extract_zip(local_zip) else 1

    destination = DOWNLOAD_DIR / "oulad.zip"
    try:
        links = find_zip_links(OFFICIAL_PAGE)
        if not links:
            print("No se encontraron enlaces ZIP en la página oficial.")
        for link in links:
            if download_candidate(link, destination):
                return 0 if extract_zip(destination) else 1
    except requests.RequestException as exc:
        print(f"No se pudo inspeccionar la página oficial: {exc}")

    print("\nDESCARGA MANUAL")
    print(f"1. Abre: {OFFICIAL_PAGE}")
    print("2. Descarga el archivo comprimido del conjunto OULAD.")
    print(f"3. Copia el ZIP a: {DOWNLOAD_DIR}")
    print("4. Ejecuta, cambiando el nombre si es necesario:")
    print(r"   python scripts\01_download_oulad.py --local-zip data/raw/downloads/oulad.zip")
    print("No se ha descargado ni extraído el conjunto automáticamente.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
