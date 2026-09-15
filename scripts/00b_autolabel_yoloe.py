"""
Fase 1/2 — Pré-anotação automática (pseudo-labels) com YOLOE-26 (open-vocabulary).

A espécie de cada foto já é conhecida (identificação "research grade" do
iNaturalist, ver 00_download_inaturalist.py). O YOLOE só precisa LOCALIZAR o
animal: recebe prompts de texto da espécie e de termos genéricos ("mammal",
"animal"), gera caixa + máscara, e a classe gravada na label vem do
iNaturalist, não do prompt. Prompts "distratores" (pássaro, pessoa, gado,
carro...) servem para que outros objetos da foto não sejam rotulados como a espécie.

Saídas (entrada do 01_prepare_dataset.py):
  data/raw/images + data/raw/labels          -> bounding box (detecção)
  data/raw_seg/images + data/raw_seg/labels  -> polygon (segmentação)
  outputs/autolabel/report.json              -> contagens e imagens descartadas
  outputs/autolabel/revisao_<classe>.jpg     -> mosaico para revisão visual pelo grupo

IMPORTANTE: são pseudo-labels geradas por modelo. Declare isso no relatório e
revise uma amostra (ex.: os mosaicos e as imagens de menor confiança) antes do treino final.

Uso:
    python scripts/00b_autolabel_yoloe.py --model models/yoloe-26l-seg.pt --device mps
"""
import argparse
import contextlib
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

# prompts que localizam cada espécie (a classe final vem do iNaturalist)
TARGET_PROMPTS = {
    "capivara": ["capybara", "rodent", "mammal", "animal"],
    "cachorro_do_mato": ["crab-eating fox", "fox", "wild dog", "mammal", "animal"],
    "tamandua_bandeira": ["giant anteater", "anteater", "mammal", "animal"],
    "tatu": ["armadillo", "mammal", "animal"],
    "lobo_guara": ["maned wolf", "wolf", "fox", "mammal", "animal"],
}

# objetos que aparecem junto nas fotos e NÃO devem virar label da espécie
DISTRACTOR_PROMPTS = ["bird", "person", "cattle", "horse", "car", "motorcycle", "cat"]
# "dog" só é distrator para espécies que não parecem cachorro
DOG_DISTRACTOR_FOR = {"capivara", "tamandua_bandeira", "tatu"}


def prompts_for(class_name):
    """Devolve (todos os prompts, conjunto dos prompts que contam como a espécie)."""
    targets = TARGET_PROMPTS[class_name]
    distractors = list(DISTRACTOR_PROMPTS)
    if class_name in DOG_DISTRACTOR_FOR:
        distractors.append("dog")
    return targets + distractors, set(targets)


def select_instances(names, confs, areas, target_names, conf_thresh=0.3, min_area=0.002):
    """
    Índices das detecções que viram label: prompt da espécie, confiança mínima
    e área mínima (fração da imagem) para descartar ruído muito pequeno.
    """
    return [
        i for i, (name, conf, area) in enumerate(zip(names, confs, areas))
        if name in target_names and conf >= conf_thresh and area >= min_area
    ]


def box_iou(a, b):
    """IoU entre caixas [x1, y1, x2, y2]."""
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedupe_instances(indices, boxes_xyxy, confs, iou_thresh=0.7):
    """
    Remove detecções repetidas do mesmo animal vindas de prompts diferentes
    (ex.: "capybara" e "mammal" na mesma região) — o YOLOE-26 não usa NMS, então
    agnostic_nms não resolve isso. Mantém a de maior confiança.
    """
    kept = []
    for i in sorted(indices, key=lambda i: confs[i], reverse=True):
        if all(box_iou(boxes_xyxy[i], boxes_xyxy[k]) < iou_thresh for k in kept):
            kept.append(i)
    return sorted(kept)


def format_bbox_line(cls_id, xywhn):
    cx, cy, w, h = (min(max(float(v), 0.0), 1.0) for v in xywhn)
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def format_polygon_line(cls_id, xyn):
    """Linha YOLO-seg; None se a máscara tiver menos de 3 pontos (polygon inválido)."""
    points = [(min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)) for x, y in xyn]
    if len(points) < 3:
        return None
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"{cls_id} {coords}"


def link_or_copy(src: Path, dest: Path):
    """Hardlink economiza disco (mesma imagem em raw e raw_seg); cai para cópia se não der."""
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def ensure_empty_output(root: Path, overwrite: bool):
    for sub in ("images", "labels"):
        d = root / sub
        if d.exists() and any(d.iterdir()):
            if not overwrite:
                sys.exit(f"[erro] {d} não está vazia. Use --overwrite para substituir as labels.")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def review_mosaic(plotted_images, out_path: Path, tile=320, cols=6):
    if not plotted_images:
        return
    tiles = []
    for img in plotted_images:
        h, w = img.shape[:2]
        scale = tile / max(h, w)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        canvas = np.full((tile, tile, 3), 255, dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        tiles.append(canvas)
    while len(tiles) % cols:
        tiles.append(np.full((tile, tile, 3), 255, dtype=np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), np.vstack(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=Path, default=Path("data/inaturalist"))
    parser.add_argument("--attribution_csv", type=Path, default=Path("data/ATRIBUICAO_imagens.csv"))
    parser.add_argument("--classes", type=Path, default=Path("config/classes.yaml"))
    parser.add_argument("--det_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seg_dir", type=Path, default=Path("data/raw_seg"))
    parser.add_argument("--report_dir", type=Path, default=Path("outputs/autolabel"))
    parser.add_argument("--model", type=str, default="models/yoloe-26l-seg.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--min_area", type=float, default=0.002, help="área mínima da caixa (fração da imagem)")
    parser.add_argument("--review_per_class", type=int, default=24)
    parser.add_argument("--device", type=str, default=None,
                        help="0 (GPU CUDA), mps (Mac Apple Silicon) ou cpu; vazio = automático")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ultralytics import YOLOE  # import aqui para os testes não carregarem torch

    names_cfg = yaml.safe_load(args.classes.read_text())["names"]
    class_ids = {name: cls_id for cls_id, name in names_cfg.items()}

    with open(args.attribution_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["classe"]].append(row["arquivo"])

    for root in (args.det_dir, args.seg_dir):
        ensure_empty_output(root, args.overwrite)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    model = YOLOE(args.model)
    rng = random.Random(args.seed)
    report = {"modelo": args.model, "conf": args.conf, "min_area": args.min_area, "classes": {}}

    for class_name, files in by_class.items():
        if class_name not in class_ids:
            print(f"[aviso] classe {class_name} não está em {args.classes}, pulando")
            continue
        cls_id = class_ids[class_name]
        all_prompts, target_names = prompts_for(class_name)
        # o Ultralytics baixa o encoder de texto (mobileclip, ~240 MB) na pasta atual;
        # rodar set_classes dentro da pasta do modelo mantém tudo em models/
        with contextlib.chdir(Path(args.model).resolve().parent):
            model.set_classes(all_prompts)

        stats = {"imagens": len(files), "rotuladas": 0, "instancias": 0,
                 "sem_deteccao": [], "confiancas": []}
        review_candidates = []
        review_idx = set(rng.sample(range(len(files)), min(args.review_per_class, len(files))))

        for idx, filename in enumerate(files):
            img_path = args.src_dir / class_name / filename
            if not img_path.exists():
                print(f"[aviso] {img_path} não encontrada")
                continue
            r = model.predict(str(img_path), conf=min(args.conf, 0.1), agnostic_nms=True,
                              device=args.device, verbose=False)[0]
            if r.boxes is None or r.masks is None or len(r.boxes) == 0:
                stats["sem_deteccao"].append(filename)
                continue

            det_names = [r.names[int(c)] for c in r.boxes.cls.tolist()]
            confs = r.boxes.conf.tolist()
            areas = [w * h for _, _, w, h in r.boxes.xywhn.tolist()]
            keep = select_instances(det_names, confs, areas, target_names, args.conf, args.min_area)
            keep = dedupe_instances(keep, r.boxes.xyxy.tolist(), confs)

            det_lines, seg_lines, kept_confs = [], [], []
            for i in keep:
                poly = format_polygon_line(cls_id, r.masks.xyn[i])
                if poly is None:
                    continue
                det_lines.append(format_bbox_line(cls_id, r.boxes.xywhn[i].tolist()))
                seg_lines.append(poly)
                kept_confs.append(round(confs[i], 3))

            if not det_lines:
                stats["sem_deteccao"].append(filename)
                continue

            stem = Path(filename).stem
            for root, lines in ((args.det_dir, det_lines), (args.seg_dir, seg_lines)):
                link_or_copy(img_path, root / "images" / filename)
                (root / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            stats["rotuladas"] += 1
            stats["instancias"] += len(det_lines)
            stats["confiancas"].append({"arquivo": filename, "min_conf": min(kept_confs)})

            if idx in review_idx:
                # só as instâncias mantidas aparecem no mosaico de revisão
                plotted = r[keep].plot() if len(keep) < len(r.boxes) else r.plot()
                review_candidates.append(plotted)

        review_mosaic(review_candidates, args.report_dir / f"revisao_{class_name}.jpg")
        stats["confiancas"].sort(key=lambda x: x["min_conf"])
        stats["menor_confianca_revisar_primeiro"] = stats.pop("confiancas")[:20]
        report["classes"][class_name] = stats
        print(f"[{class_name}] {stats['rotuladas']}/{stats['imagens']} imagens rotuladas, "
              f"{stats['instancias']} instâncias, {len(stats['sem_deteccao'])} descartadas")

    with open(args.report_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Relatório em {args.report_dir / 'report.json'}; mosaicos de revisão em {args.report_dir}")
