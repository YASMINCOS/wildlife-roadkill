"""
Fase 4 — Avaliação completa no conjunto de teste (nunca visto no treino).

Gera:
  - mAP@0.5 e mAP@0.5:0.95, precisão, recall (via Ultralytics model.val)
  - mAP das máscaras, se os pesos forem de segmentação
  - matriz de confusão (salva automaticamente pelo Ultralytics em outputs/eval/)
  - IoU médio por classe (acertos casados com as labels ground-truth)
  - exemplos de falsos positivos e falsos negativos (para o relatório)
  - opcional: figuras lado a lado caixa × máscara (--compare_boxes_masks)

Uso:
    python scripts/04_evaluate.py --weights outputs/detect/weights/best.pt \
        --data config/classes.yaml --split test
    python scripts/04_evaluate.py --weights outputs/segment/weights/best.pt \
        --data config/classes_seg.yaml --split test --compare_boxes_masks
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def compute_iou(box_a, box_b):
    """IoU entre duas caixas [x1, y1, x2, y2]. Caixas degeneradas (área <= 0) dão IoU 0."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def load_gt_boxes(label_path: Path, img_w, img_h):
    """
    Lê uma label YOLO (bbox ou polygon) e devolve [(classe, [x1, y1, x2, y2])] em pixels.
    Polygon vira a caixa envolvente, para comparar com as caixas preditas.
    """
    if not label_path.exists():
        return []
    gts = []
    for line in label_path.read_text().splitlines():
        tokens = line.split()
        if len(tokens) < 5:
            continue
        cls_id = int(float(tokens[0]))
        coords = [float(t) for t in tokens[1:]]
        if len(coords) == 4:
            cx, cy, w, h = coords
            box = [(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                   (cx + w / 2) * img_w, (cy + h / 2) * img_h]
        else:
            xs, ys = coords[0::2], coords[1::2]
            box = [min(xs) * img_w, min(ys) * img_h, max(xs) * img_w, max(ys) * img_h]
        gts.append((cls_id, box))
    return gts


def match_predictions(gts, preds, iou_thresh=0.5):
    """
    Casa predições com ground-truth da mesma classe (guloso, maior confiança primeiro).

    gts:   [(classe, caixa)]
    preds: [(classe, caixa, confiança)]
    Retorna (tp, fp, fn):
      tp = [(idx_gt, idx_pred, classe, iou)]
      fp = índices de predições sem GT correspondente (inclui classe errada)
      fn = índices de GT que nenhuma predição encontrou
    """
    order = sorted(range(len(preds)), key=lambda i: preds[i][2], reverse=True)
    matched_gt = set()
    tp, fp = [], []
    for pi in order:
        p_cls, p_box, _ = preds[pi]
        best_iou, best_gi = iou_thresh, None
        for gi, (g_cls, g_box) in enumerate(gts):
            if gi in matched_gt or g_cls != p_cls:
                continue
            iou = compute_iou(p_box, g_box)
            if iou >= best_iou:
                best_iou, best_gi = iou, gi
        if best_gi is None:
            fp.append(pi)
        else:
            matched_gt.add(best_gi)
            tp.append((best_gi, pi, p_cls, best_iou))
    fn = [gi for gi in range(len(gts)) if gi not in matched_gt]
    return tp, fp, fn


def mean_iou_per_class(tp, class_names):
    """IoU médio dos acertos por classe; None para classe sem nenhum acerto."""
    per_class = defaultdict(list)
    for _, _, cls_id, iou in tp:
        per_class[cls_id].append(iou)
    return {
        name: (sum(per_class[cls_id]) / len(per_class[cls_id]) if per_class[cls_id] else None)
        for cls_id, name in sorted(class_names.items())
    }


def list_images(split_dir):
    if not split_dir or not Path(split_dir).is_dir():
        return []
    return sorted(p for p in Path(split_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def split_by_real_format(image_paths, allowed_formats):
    """
    Separa as imagens pelo formato real do arquivo (lido pelo Pillow), não pela extensão.

    O Ultralytics descarta como "corrupt" arquivos cujo formato não está em IMG_FORMATS
    (ex.: GIF salvo com extensão .jpg, comum no iNaturalist). A análise de erros precisa
    pular os mesmos arquivos, senão conta instâncias que o model.val() não avaliou.
    Retorna (validas, ignoradas).
    """
    from PIL import Image

    validas, ignoradas = [], []
    for p in image_paths:
        try:
            with Image.open(p) as im:
                fmt = (im.format or "").lower()
        except OSError:
            fmt = ""
        (validas if fmt in allowed_formats else ignoradas).append(p)
    return validas, ignoradas


def predict_one(model, img_path, conf, device):
    """Roda o modelo em uma imagem e devolve (resultado, [(classe, caixa, confiança)])."""
    r = model.predict(source=str(img_path), conf=conf, device=device, verbose=False)[0]
    preds = []
    if r.boxes is not None:
        preds = list(zip(
            [int(c) for c in r.boxes.cls.tolist()],
            r.boxes.xyxy.tolist(),
            r.boxes.conf.tolist(),
        ))
    return r, preds


def analyze_errors(model, image_paths, out_dir: Path, conf_thresh=0.25, iou_thresh=0.5,
                   max_examples=10, device=None):
    """
    Cruza as predições com as labels ground-truth de cada imagem: calcula IoU médio
    por classe e salva as imagens com mais falsos positivos (FP) e falsos negativos (FN).
    Nas figuras, as caixas do modelo vêm do Ultralytics e as GT não detectadas
    aparecem em verde com o prefixo "FN".
    """
    from ultralytics.data.utils import img2label_paths

    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = model.names
    all_tp = []
    fp_counts, fn_counts = [], []  # (quantidade, caminho)

    for img_path in image_paths:
        r, preds = predict_one(model, img_path, conf_thresh, device)
        img_h, img_w = r.orig_shape
        label_path = Path(img2label_paths([str(img_path)])[0])
        gts = load_gt_boxes(label_path, img_w, img_h)
        tp, fp, fn = match_predictions(gts, preds, iou_thresh)
        all_tp.extend(tp)
        if fp:
            fp_counts.append((len(fp), img_path))
        if fn:
            fn_counts.append((len(fn), img_path))

    # Só re-renderiza os piores exemplos, para não guardar todas as imagens na memória
    for kind, counts in (("fp", fp_counts), ("fn", fn_counts)):
        counts.sort(key=lambda x: (-x[0], str(x[1])))
        for i, (n_err, img_path) in enumerate(counts[:max_examples]):
            r, preds = predict_one(model, img_path, conf_thresh, device)
            img_h, img_w = r.orig_shape
            gts = load_gt_boxes(Path(img2label_paths([str(img_path)])[0]), img_w, img_h)
            _, _, fn = match_predictions(gts, preds, iou_thresh)
            plotted = r.plot()
            for gi in fn:
                cls_id, (x1, y1, x2, y2) = gts[gi]
                cv2.rectangle(plotted, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(plotted, f"FN: {class_names.get(cls_id, cls_id)}", (int(x1), max(int(y1) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / f"{kind}_{i:02d}_{img_path.stem}_n{n_err}.jpg"), plotted)

    n_fp = sum(n for n, _ in fp_counts)
    n_fn = sum(n for n, _ in fn_counts)
    print(f"[análise de erros] {len(all_tp)} acertos, {n_fp} falsos positivos, {n_fn} falsos negativos "
          f"(conf >= {conf_thresh}, IoU >= {iou_thresh})")
    print(f"[análise de erros] exemplos salvos em {out_dir}")
    print("Comente manualmente cada exemplo no relatório: qual o padrão do erro "
          "(oclusão, baixa luz, espécie parecida, animal pequeno...)?")
    return {
        "iou_medio_por_classe": mean_iou_per_class(all_tp, class_names),
        "acertos": len(all_tp),
        "falsos_positivos": n_fp,
        "falsos_negativos": n_fn,
        "conf_limiar": conf_thresh,
        "iou_limiar": iou_thresh,
    }


def compare_boxes_masks(model, image_paths, out_dir: Path, conf_thresh=0.25, n_examples=6, device=None):
    """Salva figuras lado a lado: só caixas (esquerda) × só máscaras (direita)."""
    if model.task != "segment":
        print("[aviso] --compare_boxes_masks precisa de pesos de segmentação (ex.: outputs/segment/weights/best.pt)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for img_path in image_paths:
        if saved >= n_examples:
            break
        r, preds = predict_one(model, img_path, conf_thresh, device)
        if not preds:
            continue  # sem detecção as duas metades ficariam iguais
        side_by_side = cv2.hconcat([r.plot(masks=False), r.plot(boxes=False)])
        cv2.imwrite(str(out_dir / f"box_vs_mask_{img_path.stem}.jpg"), side_by_side)
        saved += 1
    print(f"[caixa x máscara] {saved} figuras salvas em {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--project", type=str, default="outputs")
    parser.add_argument("--name", type=str, default="eval")
    parser.add_argument("--conf", type=float, default=0.25, help="limiar de confiança da análise de erros")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU mínimo para contar como acerto")
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--compare_boxes_masks", action="store_true",
                        help="salva figuras caixa x máscara (pesos de segmentação)")
    parser.add_argument("--device", type=str, default=None,
                        help="0 (GPU CUDA), mps (Mac Apple Silicon) ou cpu; vazio = automático")
    args = parser.parse_args()

    # import aqui para os testes das funções puras não precisarem carregar torch
    from ultralytics import YOLO
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset

    model = YOLO(args.weights)
    # caminho absoluto: no Ultralytics 8.4+ um project relativo vai parar em ~/runs/
    out_dir = Path(args.project).resolve() / args.name

    # mAP, precisão, recall, matriz de confusão (tudo built-in do Ultralytics)
    metrics = model.val(data=args.data, split=args.split, project=str(out_dir.parent),
                        name=args.name, plots=True, exist_ok=True, device=args.device)

    summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
    }
    if model.task == "segment":
        summary.update({
            "mask_mAP50": float(metrics.seg.map50),
            "mask_mAP50-95": float(metrics.seg.map),
            "mask_precision_mean": float(metrics.seg.mp),
            "mask_recall_mean": float(metrics.seg.mr),
        })

    print("=== Resumo das métricas (conjunto de teste) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")
    print(f"\nMatriz de confusão salva automaticamente em: {out_dir}/confusion_matrix.png")
    print("Insira mAP50, mAP50-95, precisão, recall e a matriz no relatório técnico.")

    # Análise de erros — usa os caminhos já resolvidos pelo Ultralytics a partir do yaml
    data_cfg = check_det_dataset(args.data)
    image_paths = list_images(data_cfg.get(args.split))
    # Mesmo filtro de formato do model.val(), para os totais baterem com o mAP
    image_paths, ignoradas = split_by_real_format(image_paths, IMG_FORMATS)
    if ignoradas:
        print(f"[aviso] {len(ignoradas)} imagens ignoradas (formato não suportado pelo Ultralytics): "
              + ", ".join(p.name for p in ignoradas))
        summary["imagens_ignoradas"] = [p.name for p in ignoradas]
    if image_paths:
        summary["analise_erros"] = analyze_errors(
            model, image_paths, out_dir / "error_examples", conf_thresh=args.conf,
            iou_thresh=args.iou_thresh, max_examples=args.max_examples, device=args.device)
        print("[IoU médio por classe]", summary["analise_erros"]["iou_medio_por_classe"])
        if args.compare_boxes_masks:
            compare_boxes_masks(model, image_paths, out_dir / "box_vs_mask",
                                conf_thresh=args.conf, device=args.device)
    else:
        print(f"[aviso] pasta do split '{args.split}' não encontrada ({data_cfg.get(args.split)}), "
              "pulando análise de erros")

    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Resumo salvo em {out_dir / 'metrics_summary.json'}")
