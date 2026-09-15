"""Testes das funções puras de scripts/00b_autolabel_yoloe.py (sem modelo nem GPU)."""
import pytest

from conftest import SCRIPTS_DIR, load_script


@pytest.fixture(scope="module")
def autolabel():
    return load_script("00b_autolabel_yoloe.py")


def test_prompts_cobrem_todas_as_classes_do_config(autolabel):
    import yaml
    names = yaml.safe_load((SCRIPTS_DIR.parent / "config" / "classes.yaml").read_text())["names"]
    assert set(autolabel.TARGET_PROMPTS) == set(names.values())


def test_prompts_distratores_nao_contam_como_especie(autolabel):
    all_prompts, targets = autolabel.prompts_for("capivara")
    assert "bird" in all_prompts and "bird" not in targets
    assert "dog" in all_prompts  # capivara não parece cachorro


def test_dog_nao_e_distrator_para_canideos(autolabel):
    for canid in ("cachorro_do_mato", "lobo_guara"):
        all_prompts, _ = autolabel.prompts_for(canid)
        assert "dog" not in all_prompts


def test_select_instances_filtra_prompt_conf_e_area(autolabel):
    names = ["capybara", "bird", "animal", "capybara"]
    confs = [0.9, 0.95, 0.2, 0.5]
    areas = [0.3, 0.1, 0.2, 0.0001]
    keep = autolabel.select_instances(names, confs, areas, {"capybara", "animal"},
                                      conf_thresh=0.3, min_area=0.002)
    # bird é distrator, animal tem conf baixa, a última capivara é pequena demais
    assert keep == [0]


def test_select_instances_vazio(autolabel):
    assert autolabel.select_instances([], [], [], {"capybara"}) == []


def test_dedupe_remove_mesmo_animal_de_prompts_diferentes(autolabel):
    boxes = [[0, 0, 10, 10], [0, 0, 10, 9.5], [50, 50, 60, 60]]
    confs = [0.4, 0.8, 0.5]
    # caixas 0 e 1 são o mesmo animal: fica a de maior confiança (1); a 2 é outro animal
    assert autolabel.dedupe_instances([0, 1, 2], boxes, confs) == [1, 2]


def test_dedupe_mantem_animais_proximos(autolabel):
    boxes = [[0, 0, 10, 10], [6, 0, 16, 10]]   # IoU = 4/16 = 0.25
    assert autolabel.dedupe_instances([0, 1], boxes, [0.9, 0.8]) == [0, 1]


def test_dedupe_so_considera_indices_recebidos(autolabel):
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    assert autolabel.dedupe_instances([1], boxes, [0.9, 0.8]) == [1]


def test_format_bbox_line(autolabel):
    assert autolabel.format_bbox_line(3, [0.5, 0.25, 0.1, 0.2]) == "3 0.500000 0.250000 0.100000 0.200000"


def test_format_bbox_line_limita_ao_intervalo_0_1(autolabel):
    assert autolabel.format_bbox_line(0, [1.2, -0.1, 0.5, 0.5]) == "0 1.000000 0.000000 0.500000 0.500000"


def test_format_polygon_line(autolabel):
    line = autolabel.format_polygon_line(1, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.1]])
    assert line == "1 0.100000 0.200000 0.300000 0.400000 0.500000 0.100000"


def test_format_polygon_invalido(autolabel):
    assert autolabel.format_polygon_line(1, [[0.1, 0.2], [0.3, 0.4]]) is None
    assert autolabel.format_polygon_line(1, []) is None


def test_polygon_gerado_e_lido_como_polygon_pelo_01(autolabel, prepare, tmp_path):
    label = tmp_path / "a.txt"
    label.write_text(autolabel.format_polygon_line(4, [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]) + "\n")
    assert prepare.parse_label_file(label) == ([4], 1)
