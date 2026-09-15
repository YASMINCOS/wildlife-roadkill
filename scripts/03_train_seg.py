"""
Fase 3 — Segmentação de instâncias (YOLO-seg).

Requer que data/processed/segment tenha labels em formato polygon (não bbox).
Se o dataset público só tiver bounding boxes, anote as máscaras de um
subconjunto representativo (ver docs/fase1_proposta.md, seção "anotação própria")
e gere os splits com:
    python scripts/01_prepare_dataset.py --raw_dir data/raw_seg --out_dir data/processed --task segment

Uso:
    python scripts/03_train_seg.py --data config/classes_seg.yaml --epochs 100 --imgsz 640
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="config/classes_seg.yaml")
    parser.add_argument("--model", type=str, default="yolo26n-seg.pt",
                         help="checkpoint base YOLO26-seg: yolo26n/s/m-seg.pt (n = mais rápido, m = mais preciso)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default="outputs")
    parser.add_argument("--name", type=str, default="segment")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default=None,
                         help="0 (GPU CUDA), mps (Mac Apple Silicon) ou cpu; vazio = automático")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        # caminho absoluto: no Ultralytics 8.4+ um project relativo vai parar em ~/runs/
        project=str(Path(args.project).resolve()),
        name=args.name,
        patience=args.patience,
        device=args.device,
        seed=42,
        exist_ok=True,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print("Treino de segmentação concluído.")
    print(f"Melhores pesos em: {best}")
    print("\nPróximo passo sugerido (Fase 3): comparar visualmente caixas x máscaras")
    print(f"— rode scripts/04_evaluate.py --weights {best} --data {args.data} --compare_boxes_masks")
