"""Testes das funções puras de scripts/00_download_inaturalist.py (sem acesso à rede)."""
import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def download():
    return load_script("00_download_inaturalist.py")


def test_large_photo_url(download):
    url = "https://inaturalist-open-data.s3.amazonaws.com/photos/123/square.jpg"
    assert download.large_photo_url(url) == "https://inaturalist-open-data.s3.amazonaws.com/photos/123/large.jpg"


def test_large_photo_url_so_troca_o_nome_do_arquivo(download):
    url = "https://static.inaturalist.org/photos/square_dir/456/square.jpeg"
    assert download.large_photo_url(url).endswith("/square_dir/456/large.jpeg")


def test_pick_photo_ignora_licenca_nao_permitida(download):
    obs = {"photos": [
        {"id": 1, "license_code": None, "url": "u1"},             # todos os direitos reservados
        {"id": 2, "license_code": "cc-by-nd", "url": "u2"},       # sem derivações: fora da lista
        {"id": 3, "license_code": "CC-BY", "url": "u3"},
    ]}
    assert download.pick_photo(obs)["id"] == 3


def test_pick_photo_sem_foto_valida(download):
    assert download.pick_photo({"photos": [{"id": 1, "license_code": "cc-by-nd", "url": "u"}]}) is None
    assert download.pick_photo({"photos": []}) is None
    assert download.pick_photo({}) is None


def test_attribution_row(download):
    obs = {"id": 99, "uri": "https://www.inaturalist.org/observations/99",
           "observed_on": "2024-05-01", "place_guess": "Miranda, MS, Brasil"}
    photo = {"id": 7, "license_code": "cc-by-nc", "attribution": "(c) fulano, some rights reserved (CC BY-NC)"}
    row = download.attribution_row(obs, photo, "tatu", "inat_7.jpg")
    assert row["especie"] == "Euphractus sexcinctus"
    assert row["licenca"] == "CC-BY-NC"
    assert row["observacao_url"].endswith("/99")
    assert set(row) == set(download.CSV_FIELDS)


def test_attribution_row_sem_uri(download):
    photo = {"id": 7, "license_code": "cc0"}
    row = download.attribution_row({"id": 5}, photo, "capivara", "inat_7.jpg")
    assert row["observacao_url"] == "https://www.inaturalist.org/observations/5"
    assert row["data_observacao"] == "" and row["local"] == ""


def test_classes_batem_com_config(download):
    import yaml
    from conftest import SCRIPTS_DIR
    names = yaml.safe_load((SCRIPTS_DIR.parent / "config" / "classes.yaml").read_text())["names"]
    assert list(download.SPECIES) == [names[i] for i in sorted(names)]
