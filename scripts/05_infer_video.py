"""
Fase 4/Item 5 — Inferência em vídeo real do cenário (mínimo 30s, requisito obrigatório).

Bônus: rastreamento de objetos (ByteTrack) entre frames — soma até +0,5 na nota.

Uso (sem tracking, obrigatório):
    python scripts/05_infer_video.py --weights outputs/segment/weights/best.pt \
        --source data/video_teste.mp4clear && ver scripts/02_train_detect.py 33 52


Uso (com tracking, bônus):
    python scripts/05_infer_video.py --weights outputs/segment/weights/best.pt \
        --source data/video_teste.mp4 --track

--source também aceita índice de webcam (ex.: 0) ou URL de câmera (rtsp://...).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="caminho do vídeo (>= 30s)")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--track", action="store_true", help="ativa ByteTrack (bônus)")
    parser.add_argument("--project", type=str, default="outputs")
    parser.add_argument("--name", type=str, default="video_inference")
    parser.add_argument("--device", type=str, default=None,
                        help="0 (GPU CUDA), mps (Mac Apple Silicon) ou cpu; vazio = automático")
    args = parser.parse_args()

    # Valida o arquivo antes de carregar o modelo; webcam e URL não são arquivos locais
    is_local_file = not (args.source.isdigit() or "://" in args.source)
    if is_local_file and not Path(args.source).exists():
        raise FileNotFoundError(
            f"{args.source} não encontrado. Grave um vídeo real (>=30s) do cenário "
            "(ex.: trecho de estrada/parque) e salve em data/."
        )

    model = YOLO(args.weights)

    common = dict(
        source=args.source,
        conf=args.conf,
        save=True,
        # caminho absoluto: no Ultralytics 8.4+ um project relativo vai parar em ~/runs/
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        device=args.device,
        # stream=True devolve um gerador: o vídeo é processado frame a frame sem
        # acumular todos os resultados (e máscaras) na memória
        stream=True,
    )
    if args.track:
        results = model.track(tracker="bytetrack.yaml", **common)
    else:
        results = model.predict(**common)

    n_frames = sum(1 for _ in results)  # consome o gerador (é aqui que a inferência roda)

    if args.track:
        print(f"Inferência COM tracking (ByteTrack) concluída em {n_frames} frames — critério de bônus atendido.")
    else:
        print(f"Inferência sem tracking concluída em {n_frames} frames.")

    print(f"Vídeo com overlay de detecção/segmentação salvo em: {model.predictor.save_dir}")
    print("Use este vídeo no Item 4 (vídeo-pitch) e no Item 5 do barema (Aplicação em vídeo).")
