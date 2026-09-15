"""
Fase 2 — Baseline de detecção (fine-tuning YOLO).

Uso (no Colab, com GPU):
    python scripts/02_train_detect.py --data config/classes.yaml --epochs 100 --imgsz 640 --batch 16

Uso (teste curto local, Mac Apple Silicon):
    python scripts/02_train_detect.py --data config/classes.yaml --epochs 5 --device mps

Registra curvas de treino automaticamente em outputs/detect/ (via Ultralytics).
Documente aqui os hiperparâmetros finais usados no relatório técnico.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="config/classes.yaml")
    parser.add_argument("--model", type=str, default="yolo26n.pt",
                         help="checkpoint base YOLO26: yolo26n/s/m.pt (n = mais rápido, m = mais preciso)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default="outputs")
    parser.add_argument("--name", type=str, default="detect")
    parser.add_argument("--patience", type=int, default=20, help="early stopping")
    parser.add_argument("--device", type=str, default=None,
                         help="0 (GPU CUDA), mps (Mac Apple Silicon) ou cpu; vazio = automático")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        # caminho absoluto: no Ultralytics 8.4+ um project relativo vai parar em
        # ~/runs/<task>/outputs, e não em outputs/ dentro do repositório
        project=str(Path(args.project).resolve()),
        name=args.name,
        patience=args.patience,
        device=args.device,
        # augmentation — ajuste conforme a EDA (ex.: mais mosaic se dataset pequeno)
        mosaic=1.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5,
        seed=42,
        exist_ok=True,
    )

    print("Treino concluído.")
    print(f"Melhores pesos em: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")
    print("Hiperparâmetros usados (documentar no relatório):")
    print(vars(args))
