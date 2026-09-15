"""Testes das funções puras de scripts/04_evaluate.py (não precisam de GPU, pesos nem dataset)."""
import pytest


# ---------------------------------------------------------------- compute_iou

def test_iou_caixas_identicas(evaluate):
    assert evaluate.compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_caixas_disjuntas(evaluate):
    assert evaluate.compute_iou([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0


def test_iou_caixas_so_encostam(evaluate):
    assert evaluate.compute_iou([0, 0, 1, 1], [1, 0, 2, 1]) == 0.0


def test_iou_sobreposicao_parcial(evaluate):
    # interseção 1x1 = 1; união = 4 + 4 - 1 = 7
    assert evaluate.compute_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)


def test_iou_caixa_contida(evaluate):
    assert evaluate.compute_iou([0, 0, 4, 4], [1, 1, 3, 3]) == pytest.approx(4 / 16)


def test_iou_simetrico(evaluate):
    a, b = [0, 0, 5, 3], [2, 1, 7, 6]
    assert evaluate.compute_iou(a, b) == pytest.approx(evaluate.compute_iou(b, a))


@pytest.mark.parametrize("box_a, box_b", [
    ([0, 0, 0, 0], [0, 0, 0, 0]),   # duas caixas de área zero
    ([2, 2, 0, 0], [0, 0, 2, 2]),   # caixa invertida (x2 < x1)
])
def test_iou_caixas_degeneradas(evaluate, box_a, box_b):
    assert evaluate.compute_iou(box_a, box_b) == 0.0


# ---------------------------------------------------------------- load_gt_boxes

def test_gt_bbox_normalizada_para_pixels(evaluate, tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("2 0.5 0.5 0.2 0.4\n")
    [(cls_id, box)] = evaluate.load_gt_boxes(label, img_w=100, img_h=200)
    assert cls_id == 2
    assert box == pytest.approx([40, 60, 60, 140])


def test_gt_polygon_vira_caixa_envolvente(evaluate, tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("1 0.1 0.2 0.3 0.4 0.2 0.1\n")
    [(cls_id, box)] = evaluate.load_gt_boxes(label, img_w=100, img_h=100)
    assert cls_id == 1
    assert box == pytest.approx([10, 10, 30, 40])


def test_gt_label_inexistente(evaluate, tmp_path):
    assert evaluate.load_gt_boxes(tmp_path / "nao_existe.txt", 100, 100) == []


# ---------------------------------------------------------------- match_predictions

def test_match_acerto(evaluate):
    tp, fp, fn = evaluate.match_predictions([(0, [0, 0, 10, 10])], [(0, [0, 0, 10, 10], 0.9)])
    assert len(tp) == 1 and tp[0][2] == 0 and tp[0][3] == pytest.approx(1.0)
    assert fp == [] and fn == []


def test_match_classe_errada_conta_fp_e_fn(evaluate):
    tp, fp, fn = evaluate.match_predictions([(0, [0, 0, 10, 10])], [(1, [0, 0, 10, 10], 0.9)])
    assert tp == [] and fp == [0] and fn == [0]


def test_match_iou_abaixo_do_limiar(evaluate):
    tp, fp, fn = evaluate.match_predictions([(0, [0, 0, 2, 2])], [(0, [1, 1, 3, 3], 0.9)], iou_thresh=0.5)
    assert tp == [] and fp == [0] and fn == [0]


def test_match_duas_predicoes_mesma_gt(evaluate):
    gts = [(0, [0, 0, 10, 10])]
    preds = [(0, [0, 0, 10, 9], 0.3), (0, [0, 0, 10, 10], 0.8)]
    tp, fp, fn = evaluate.match_predictions(gts, preds)
    assert [t[1] for t in tp] == [1]   # a de maior confiança fica com a GT
    assert fp == [0] and fn == []


def test_match_sem_gt_nem_predicao(evaluate):
    assert evaluate.match_predictions([], []) == ([], [], [])


# ---------------------------------------------------------------- split_by_real_format

def test_formato_real_ignora_gif_com_extensao_jpg(evaluate, tmp_path):
    from PIL import Image

    jpg = tmp_path / "ok.jpg"
    Image.new("RGB", (8, 8)).save(jpg, format="JPEG")
    gif_disfarcado = tmp_path / "gif.jpg"
    Image.new("RGB", (8, 8)).save(gif_disfarcado, format="GIF")
    validas, ignoradas = evaluate.split_by_real_format([jpg, gif_disfarcado], {"jpg", "jpeg", "png"})
    assert validas == [jpg] and ignoradas == [gif_disfarcado]


def test_formato_real_ignora_arquivo_ilegivel(evaluate, tmp_path):
    lixo = tmp_path / "quebrada.jpg"
    lixo.write_bytes(b"nao e imagem")
    assert evaluate.split_by_real_format([lixo], {"jpeg"}) == ([], [lixo])


# ---------------------------------------------------------------- mean_iou_per_class

def test_iou_medio_por_classe(evaluate):
    names = {0: "capivara", 1: "tatu"}
    tp = [(0, 0, 0, 0.6), (1, 1, 0, 0.8)]
    assert evaluate.mean_iou_per_class(tp, names) == {"capivara": pytest.approx(0.7), "tatu": None}
