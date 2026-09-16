#!/usr/bin/env bash
# Demonstração para o vídeo-pitch: testes + detecção + segmentação + métricas + erros + vídeo.
#
# Uso (gravação — pausa entre etapas e abre as imagens/vídeo):
#     bash scripts/demo_video.sh
# Verificação rápida, sem pausas e sem abrir janelas:
#     PAUSA=0 ABRIR=0 bash scripts/demo_video.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # sempre a partir da raiz do repositório

PAUSA=${PAUSA:-1}
ABRIR=${ABRIR:-1}
PY=.venv/bin/python
YOLO=.venv/bin/yolo
IMG=data/processed/segment/images/test/inat_343605085.jpg   # tamanduá à noite (conjunto de teste)

etapa() { echo; echo "=================== $1 ==================="; }
pausa() { if [ "$PAUSA" = 1 ]; then read -rp ">> Enter para continuar..." _; fi; }
abrir() { if [ "$ABRIR" = 1 ]; then open "$@"; else echo "(abriria: $*)"; fi; }

etapa "0. Conferindo arquivos necessários"
for f in "$PY" "$YOLO" outputs/detect/weights/best.pt outputs/segment/weights/best.pt "$IMG" \
         outputs/eval/metrics_summary.json outputs/eval_seg/metrics_summary.json \
         outputs/video_inference/video_teste.mp4 docs/figuras/box_vs_mask_tamandua.jpg; do
    if [ -e "$f" ]; then echo "ok     $f"; else echo "FALTA  $f"; exit 1; fi
done
# MPS no Mac Apple Silicon; senão CPU
DEVICE=$("$PY" -c "import torch; print('mps' if torch.backends.mps.is_available() else 'cpu')")
echo "device: $DEVICE"
pausa

etapa "1. Testes automáticos (pytest)"
"$PY" -m pytest -q
pausa

etapa "2. Detecção — caixa + espécie + confiança"
"$YOLO" predict model=outputs/detect/weights/best.pt source="$IMG" project="$PWD/outputs" \
    name=demo_detect exist_ok=True device="$DEVICE" 2>&1 | grep -E "image 1/1|Results saved"
abrir outputs/demo_detect/inat_343605085.jpg
pausa

etapa "3. Segmentação — mesma foto, com máscara"
"$YOLO" predict model=outputs/segment/weights/best.pt source="$IMG" project="$PWD/outputs" \
    name=demo_seg exist_ok=True device="$DEVICE" 2>&1 | grep -E "image 1/1|Results saved"
abrir outputs/demo_seg/inat_343605085.jpg
pausa

etapa "4. Caixa (esquerda) x máscara (direita)"
abrir docs/figuras/box_vs_mask_tamandua.jpg
pausa

etapa "5. Métricas no conjunto de teste"
"$PY" - <<'EOF'
import json
det = json.load(open("outputs/eval/metrics_summary.json"))
seg = json.load(open("outputs/eval_seg/metrics_summary.json"))
print(f"{'Modelo':<28}{'mAP@0.5':>9}{'mAP@.5:.95':>12}{'Precisão':>10}{'Recall':>8}")
print(f"{'Detecção (caixas)':<28}{det['mAP50']:>9.3f}{det['mAP50-95']:>12.3f}{det['precision_mean']:>10.3f}{det['recall_mean']:>8.3f}")
print(f"{'Segmentação (máscaras)':<28}{seg['mask_mAP50']:>9.3f}{seg['mask_mAP50-95']:>12.3f}"
      f"{seg['mask_precision_mean']:>10.3f}{seg['mask_recall_mean']:>8.3f}")
EOF
abrir docs/figuras/segment_confusion_matrix_normalized.png
pausa

etapa "6. Exemplos de erro (falso positivo e falso negativo)"
abrir docs/figuras/fp1_cachorro_noite.jpg docs/figuras/fn1_tamandua_armadilha.jpg
pausa

etapa "7. Vídeo real com segmentação + rastreamento (ByteTrack)"
abrir outputs/video_inference/video_teste.mp4
echo
echo "Demonstração concluída."
