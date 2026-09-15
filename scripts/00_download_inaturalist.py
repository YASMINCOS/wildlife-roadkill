"""
Fase 1/2 — Download de imagens reais das 5 espécies-alvo no iNaturalist.

Baixa fotos de observações "research grade" (identificação confirmada pela
comunidade) com licença aberta, uma foto por observação — várias fotos da
mesma observação são quase idênticas e poderiam cair em splits diferentes.
A atribuição de cada imagem (autor, licença, link) é gravada em
data/ATRIBUICAO_imagens.csv, exigência das licenças Creative Commons.

Uso:
    python scripts/00_download_inaturalist.py --out_dir data/inaturalist --per_class 150
"""
import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "wildlife-roadkill-cv/1.0 (projeto academico de visao computacional)"

# nome da classe (config/classes.yaml) -> (id do táxon no iNaturalist, nome científico)
SPECIES = {
    "capivara": (74442, "Hydrochoerus hydrochaeris"),
    "cachorro_do_mato": (42087, "Cerdocyon thous"),
    "tamandua_bandeira": (47107, "Myrmecophaga tridactyla"),
    "tatu": (47083, "Euphractus sexcinctus"),
    "lobo_guara": (42091, "Chrysocyon brachyurus"),
}

# licenças que permitem uso acadêmico com atribuição
ALLOWED_LICENSES = ("cc0", "cc-by", "cc-by-sa", "cc-by-nc")

CSV_FIELDS = ["arquivo", "classe", "especie", "licenca", "atribuicao",
              "observacao_url", "foto_id", "data_observacao", "local"]


def large_photo_url(url: str) -> str:
    """A API devolve a miniatura (square, 75px); troca pela versão large (até 1024px)."""
    return url.replace("/square.", "/large.")


def pick_photo(observation, allowed=ALLOWED_LICENSES):
    """Primeira foto da observação com licença permitida, ou None."""
    for photo in observation.get("photos") or []:
        if (photo.get("license_code") or "").lower() in allowed and photo.get("url"):
            return photo
    return None


def attribution_row(observation, photo, class_name, filename):
    return {
        "arquivo": filename,
        "classe": class_name,
        "especie": SPECIES[class_name][1],
        "licenca": photo["license_code"].upper(),
        "atribuicao": photo.get("attribution", ""),
        "observacao_url": observation.get("uri") or f"https://www.inaturalist.org/observations/{observation['id']}",
        "foto_id": photo["id"],
        "data_observacao": observation.get("observed_on") or "",
        "local": observation.get("place_guess") or "",
    }


def http_get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as exc:  # rede instável: espera e tenta de novo
            if attempt == retries - 1:
                raise
            print(f"[aviso] falha em {url} ({exc}), tentando de novo")
            time.sleep(5 * (attempt + 1))


def fetch_observations(taxon_id, id_below=None, per_page=200):
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "photo_license": ",".join(ALLOWED_LICENSES),
        "photos": "true",
        # anotação "Evidence of Presence = Organism": sem ela vêm também fotos só
        # de fezes, pegadas e ossos, que não mostram o animal
        "term_id": 22,
        "term_value_id": 24,
        "per_page": per_page,
        "order_by": "id",
        "order": "desc",
    }
    if id_below:
        params["id_below"] = id_below
    return json.loads(http_get(f"{API_URL}?{urllib.parse.urlencode(params)}"))["results"]


def download_class(class_name, per_class, out_dir: Path):
    taxon_id = SPECIES[class_name][0]
    class_dir = out_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    id_below = None
    while len(rows) < per_class:
        observations = fetch_observations(taxon_id, id_below)
        time.sleep(1)  # limite recomendado pela API do iNaturalist: ~1 requisição/s
        if not observations:
            print(f"[aviso] {class_name}: só {len(rows)} imagens disponíveis")
            break
        id_below = observations[-1]["id"]
        for obs in observations:
            photo = pick_photo(obs)
            if photo is None:
                continue
            filename = f"inat_{photo['id']}.jpg"
            dest = class_dir / filename
            if not dest.exists():  # permite retomar um download interrompido
                try:
                    dest.write_bytes(http_get(large_photo_url(photo["url"])))
                except Exception as exc:
                    print(f"[aviso] não foi possível baixar a foto {photo['id']}: {exc}")
                    continue
            rows.append(attribution_row(obs, photo, class_name, filename))
            if len(rows) >= per_class:
                break
        print(f"[{class_name}] {len(rows)}/{per_class}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("data/inaturalist"))
    parser.add_argument("--per_class", type=int, default=150)
    parser.add_argument("--attribution_csv", type=Path, default=Path("data/ATRIBUICAO_imagens.csv"))
    args = parser.parse_args()

    args.attribution_csv.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for class_name in SPECIES:
        all_rows.extend(download_class(class_name, args.per_class, args.out_dir))
        # regrava o CSV a cada classe: se o processo for interrompido, as
        # atribuições das classes já baixadas não se perdem
        with open(args.attribution_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)

    if not all_rows:
        sys.exit("[erro] nenhuma imagem baixada — verifique a conexão com api.inaturalist.org")
    print(f"{len(all_rows)} imagens em {args.out_dir}; atribuições em {args.attribution_csv}")
