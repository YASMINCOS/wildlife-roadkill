"""
Fase 1/2 — Preparação do dataset.

Organiza um export bruto (ex.: baixado do Roboflow Universe, já em formato
YOLO) em splits fixos train/val/test com seed reprodutível, e roda uma EDA
básica (contagem por classe, resolução das imagens, desbalanceamento).

Layouts aceitos em --raw_dir (a label de cada imagem é procurada trocando a
última pasta "images" do caminho por "labels", mesma regra do Ultralytics):
    raw/images/*.jpg              + raw/labels/*.txt               (plano)
    raw/train/images/*.jpg        + raw/train/labels/*.txt         (export Roboflow)
    raw/images/train/*.jpg        + raw/labels/train/*.txt         (layout Ultralytics)
Se o export já vier dividido, as imagens são reunidas e re-divididas com a
seed informada, para o split ser o mesmo em qualquer máquina do grupo.

Uso:
    python scripts/01_prepare_dataset.py --raw_dir data/raw --out_dir data/processed --seed 42
    python scripts/01_prepare_dataset.py --raw_dir data/raw_seg --out_dir data/processed --task segment
"""
import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # salva figuras sem precisar de display (Colab/terminal)
import matplotlib.pyplot as plt
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

CLASS_NAMES = {0: "capivara", 1: "cachorro_do_mato", 2: "tamandua_bandeira",
               3: "tatu", 4: "lobo_guara"}


def label_path_for(img_path: Path) -> Path:
    """Troca a última pasta "images" do caminho por "labels" e a extensão por .txt."""
    parts = list(img_path.parts)
    if "images" not in parts[:-1]:
        raise ValueError(f"imagem fora de uma pasta 'images': {img_path}")
    idx = len(parts) - 2 - parts[-2::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def list_image_label_pairs(raw_dir: Path):
    """Lista pares (imagem, label) em qualquer um dos layouts descritos no topo do arquivo."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"pasta {raw_dir} não existe")

    image_paths = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and "images" in p.relative_to(raw_dir).parts[:-1]
    )

    pairs = []
    seen_names = set()
    for img_path in image_paths:
        label_path = label_path_for(img_path)
        if not label_path.exists():
            print(f"[aviso] sem label para {img_path.relative_to(raw_dir)}, pulando")
            continue
        # Splits de origem diferentes podem ter o mesmo nome de arquivo; como os
        # splits de saída são planos, o segundo sobrescreveria o primeiro.
        if img_path.name in seen_names:
            print(f"[aviso] nome de imagem duplicado {img_path.name}, pulando")
            continue
        seen_names.add(img_path.name)
        pairs.append((img_path, label_path))
    return pairs


def split_dataset(pairs, seed=42, ratios=(0.7, 0.15, 0.15)):
    """Divide em train/val/test de forma determinística. Não altera a lista recebida."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios devem somar 1, recebido {ratios}")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def parse_label_file(label_path: Path):
    """
    Lê uma label YOLO e devolve (ids de classe, nº de linhas em polygon).
    Linha bbox: "cls cx cy w h" (5 valores). Linha polygon: "cls x1 y1 x2 y2 ..." (>5).
    """
    class_ids = []
    n_polygon = 0
    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        try:
            # alguns exports gravam o id como "0.0"
            class_ids.append(int(float(tokens[0])))
        except ValueError:
            print(f"[aviso] linha inválida em {label_path.name}:{line_no}: {line!r}")
            continue
        if len(tokens) > 5:
            n_polygon += 1
    return class_ids, n_polygon


def check_raw_class_order(raw_dir: Path, class_names):
    """Avisa se o data.yaml do export tem outra ordem de classes (ids trocados)."""
    for yaml_path in (raw_dir / "data.yaml", raw_dir / "data.yml"):
        if not yaml_path.exists():
            continue
        names = (yaml.safe_load(yaml_path.read_text()) or {}).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        expected = [class_names[i] for i in sorted(class_names)]
        if names and list(names) != expected:
            print(f"[aviso] ordem de classes em {yaml_path} é {list(names)}, mas "
                  f"config/classes.yaml espera {expected}. Os ids das labels ficariam "
                  "trocados — reexporte com a mesma ordem ou remapeie as labels.")
        return


def copy_split(split_name, pairs, out_dir: Path):
    img_out = out_dir / "images" / split_name
    lbl_out = out_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img_path, label_path in pairs:
        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(label_path, lbl_out / label_path.name)


def run_eda(pairs, class_names, out_dir: Path, task="detect"):
    """Gera contagem por classe e distribuição de resolução — insumo para o relatório."""
    instance_counts = Counter()
    image_counts = Counter()
    resolutions = []
    n_lines = 0
    n_polygon = 0
    unknown_ids = Counter()

    for img_path, label_path in pairs:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            resolutions.append((w, h))
        else:
            print(f"[aviso] não foi possível ler {img_path.name}")
        class_ids, polys = parse_label_file(label_path)
        n_lines += len(class_ids)
        n_polygon += polys
        for cls_id in class_ids:
            if cls_id not in class_names:
                unknown_ids[cls_id] += 1
            instance_counts[cls_id] += 1
        for cls_id in set(class_ids):
            image_counts[cls_id] += 1

    if unknown_ids:
        print(f"[aviso] ids de classe fora de config/classes.yaml: {dict(unknown_ids)}")
    if task == "segment" and n_lines and n_polygon < n_lines:
        print(f"[aviso] {n_lines - n_polygon} de {n_lines} anotações não são polygon — "
              "YOLO-seg precisa de máscaras, não só bounding box.")

    # Gráfico de contagem por classe (evidencia desbalanceamento); classes sem
    # nenhuma anotação aparecem com zero em vez de sumirem do gráfico
    ids = sorted(set(class_names) | set(instance_counts))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([class_names.get(i, str(i)) for i in ids], [instance_counts[i] for i in ids])
    ax.set_title("Instâncias anotadas por classe")
    ax.set_ylabel("Contagem")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "eda_class_distribution.png", dpi=150)
    plt.close(fig)

    summary = {
        "total_imagens": len(pairs),
        "instancias_por_classe": {class_names.get(i, str(i)): instance_counts[i] for i in ids},
        "imagens_por_classe": {class_names.get(i, str(i)): image_counts[i] for i in ids},
    }
    print(f"[EDA] distribuição de classes salva em {out_dir / 'eda_class_distribution.png'}")
    print(f"[EDA] total de imagens: {len(pairs)}")
    print(f"[EDA] instâncias por classe: {summary['instancias_por_classe']}")
    print(f"[EDA] imagens por classe: {summary['imagens_por_classe']}")
    if resolutions:
        widths, heights = zip(*resolutions)
        summary["resolucao_media"] = [round(sum(widths) / len(widths)), round(sum(heights) / len(heights))]
        print(f"[EDA] resolução média: {summary['resolucao_media'][0]}x{summary['resolucao_media'][1]}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["detect", "segment"], default="detect",
                        help="define a subpasta de saída (detect -> classes.yaml, segment -> classes_seg.yaml)")
    args = parser.parse_args()

    pairs = list_image_label_pairs(args.raw_dir)
    print(f"Encontrados {len(pairs)} pares imagem/label em {args.raw_dir}")
    if not pairs:
        sys.exit(f"[erro] nenhum par imagem/label em {args.raw_dir}. Extraia o export "
                 "YOLO do Roboflow lá (pastas images/ e labels/) e rode de novo.")

    check_raw_class_order(args.raw_dir, CLASS_NAMES)
    summary = run_eda(pairs, CLASS_NAMES, args.out_dir / "eda", task=args.task)

    # Remove splits de uma execução anterior: sem isso, rodar de novo com outra
    # seed deixaria a mesma imagem em train e test ao mesmo tempo (vazamento)
    task_dir = args.out_dir / args.task
    for sub in ("images", "labels"):
        if (task_dir / sub).exists():
            print(f"[split] removendo splits anteriores em {task_dir / sub}")
            shutil.rmtree(task_dir / sub)

    splits = split_dataset(pairs, seed=args.seed)
    summary["seed"] = args.seed
    summary["splits"] = {}
    for split_name, split_pairs in splits.items():
        copy_split(split_name, split_pairs, task_dir)
        summary["splits"][split_name] = len(split_pairs)
        print(f"[split] {split_name}: {len(split_pairs)} imagens")
        if not split_pairs:
            print(f"[aviso] split {split_name} ficou vazio — dataset pequeno demais para 70/15/15")

    with open(args.out_dir / "eda" / f"eda_summary_{args.task}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Concluído. Dataset em {task_dir} (ver config/classes{'_seg' if args.task == 'segment' else ''}.yaml).")
