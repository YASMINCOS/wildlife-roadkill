"""Testes das funções puras de scripts/01_prepare_dataset.py (não precisam de GPU nem dataset)."""
from pathlib import Path

import pytest


def _touch(path: Path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------- split_dataset

def test_split_proporcoes_70_15_15(prepare):
    splits = prepare.split_dataset(list(range(100)), seed=42)
    assert [len(splits[s]) for s in ("train", "val", "test")] == [70, 15, 15]


def test_split_sem_sobreposicao_e_sem_perda(prepare):
    items = list(range(57))
    splits = prepare.split_dataset(items, seed=7)
    train, val, test = (set(splits[s]) for s in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(items)


def test_split_reprodutivel_com_mesma_seed(prepare):
    items = list(range(50))
    assert prepare.split_dataset(items, seed=42) == prepare.split_dataset(items, seed=42)


def test_split_muda_com_outra_seed(prepare):
    items = list(range(50))
    assert prepare.split_dataset(items, seed=1)["train"] != prepare.split_dataset(items, seed=2)["train"]


def test_split_nao_altera_lista_original(prepare):
    items = list(range(20))
    prepare.split_dataset(items, seed=42)
    assert items == list(range(20))


def test_split_lista_vazia(prepare):
    assert prepare.split_dataset([], seed=42) == {"train": [], "val": [], "test": []}


def test_split_dataset_pequeno_nao_perde_itens(prepare):
    splits = prepare.split_dataset([1, 2, 3], seed=42)
    assert sum(len(v) for v in splits.values()) == 3


def test_split_ratios_invalidos(prepare):
    with pytest.raises(ValueError):
        prepare.split_dataset([1, 2, 3], ratios=(0.8, 0.15, 0.15))


# ---------------------------------------------------------------- label_path_for

@pytest.mark.parametrize("img, expected", [
    ("raw/images/a.jpg", "raw/labels/a.txt"),                     # plano
    ("raw/train/images/a.JPG", "raw/train/labels/a.txt"),         # export Roboflow
    ("raw/images/train/a.png", "raw/labels/train/a.txt"),         # layout Ultralytics
    ("x/images/proj/images/a.jpg", "x/images/proj/labels/a.txt"),  # troca só a última pasta
])
def test_label_path_for(prepare, img, expected):
    assert prepare.label_path_for(Path(img)) == Path(expected)


def test_label_path_for_imagem_fora_de_images(prepare):
    with pytest.raises(ValueError):
        prepare.label_path_for(Path("raw/fotos/images.jpg"))


# ---------------------------------------------------------------- list_image_label_pairs

def test_lista_pares_layout_plano(prepare, tmp_path):
    _touch(tmp_path / "images" / "a.jpg")
    _touch(tmp_path / "labels" / "a.txt", "0 0.5 0.5 0.1 0.1")
    _touch(tmp_path / "images" / "sem_label.png")   # imagem sem label correspondente
    _touch(tmp_path / "images" / ".DS_Store")        # arquivo que não é imagem
    _touch(tmp_path / "images" / "notas.txt")
    pairs = prepare.list_image_label_pairs(tmp_path)
    assert pairs == [(tmp_path / "images" / "a.jpg", tmp_path / "labels" / "a.txt")]


def test_lista_pares_export_roboflow(prepare, tmp_path):
    for split, name in (("train", "a"), ("valid", "b"), ("test", "c")):
        _touch(tmp_path / split / "images" / f"{name}.jpg")
        _touch(tmp_path / split / "labels" / f"{name}.txt")
    pairs = prepare.list_image_label_pairs(tmp_path)
    assert sorted(img.name for img, _ in pairs) == ["a.jpg", "b.jpg", "c.jpg"]


def test_lista_pares_ignora_nome_duplicado(prepare, tmp_path):
    for split in ("train", "valid"):
        _touch(tmp_path / split / "images" / "a.jpg")
        _touch(tmp_path / split / "labels" / "a.txt")
    assert len(prepare.list_image_label_pairs(tmp_path)) == 1


def test_lista_pares_pasta_vazia(prepare, tmp_path):
    assert prepare.list_image_label_pairs(tmp_path) == []


def test_lista_pares_pasta_inexistente(prepare, tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare.list_image_label_pairs(tmp_path / "nao_existe")


# ---------------------------------------------------------------- parse_label_file

def test_parse_label_bbox_polygon_e_linhas_ruins(prepare, tmp_path):
    label = tmp_path / "a.txt"
    label.write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "\n"
        "3.0 0.1 0.1 0.2 0.1 0.2 0.2\n"   # polygon com id gravado como float
        "abc 0.1 0.1 0.1 0.1\n"           # linha inválida é ignorada
    )
    class_ids, n_polygon = prepare.parse_label_file(label)
    assert class_ids == [0, 3]
    assert n_polygon == 1


def test_parse_label_vazia(prepare, tmp_path):
    label = tmp_path / "vazia.txt"
    label.write_text("")
    assert prepare.parse_label_file(label) == ([], 0)
