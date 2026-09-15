# Relatório Técnico — Sistema de Detecção e Segmentação de Fauna Silvestre em Rodovias

**Grupo:** _[nomes completos]_ · **Disciplina:** Visão Computacional e Reconhecimento de Padrões
**Repositório:** _[link do GitHub]_ · **Vídeo-pitch:** _[link]_

> Este template segue as 6 seções pedidas no edital (Item 1 da entrega) e os
> critérios do barema. Preencha cada seção com os resultados reais do grupo;
> os comentários em itálico são orientações, remova-os na versão final.
> Declare aqui o uso de IA generativa: _[onde e como foi usada, ex.: "Claude
> foi usado para estruturar o repositório, templates de documentação e
> revisão de código; anotação, treino e análise de resultados foram feitos
> pelo grupo"]_.

---

## 1. Problema e cenário
_(peso 10% — clareza do cenário, qualidade e origem dos dados, EDA, splits corretos)_

- Contexto do problema (atropelamento de fauna em rodovias brasileiras) e
  motivação real (dados do CBEE/UFLA).
- Cenário escolhido: Cidades Inteligentes — variante fauna silvestre.
- Por que essas 5 espécies-alvo foram escolhidas.

## 2. Dataset e EDA
_(parte do critério 1 — 10%)_

- Fontes usadas (ver `data/DATASET_SOURCES.md`), com citação completa.
- Tabela: nº de imagens por classe, antes e depois do split.
- Gráfico de distribuição de classes (`data/processed/eda/eda_class_distribution.png`)
  — comente o desbalanceamento observado.
- Resolução média das imagens, condições de luz/ângulo observadas.
- Estratégia de split (70/15/15, seed=42) e por que ela é reprodutível.

## 3. Metodologia
_(critérios 2 e 3 — 25% + 20%)_

### 3.1 Detecção
- Modelo base (YOLOv8n/s/m) e por que foi escolhido.
- Hiperparâmetros finais: épocas, imgsz, batch, augmentation (mosaic, hsv, flip).
- Curvas de treino (loss, mAP por época) — anexar imagens de `outputs/detect/`.

### 3.2 Segmentação
- Modelo usado (YOLO-seg) e como as máscaras foram obtidas (anotação própria
  em CVAT, subconjunto anotado).
- Comparação visual caixa × máscara: **o que a segmentação revela que a
  detecção não mostra?** (ex.: contorno exato do animal parcialmente oculto
  por vegetação, estimativa melhor de área ocupada na pista).

## 4. Resultados e avaliação
_(critério 4 — 20%)_

- Tabela de métricas no conjunto de teste: mAP@0.5, mAP@0.5:0.95, precisão,
  recall (de `outputs/eval/metrics_summary.json`).
- IoU médio por classe.
- Matriz de confusão (`outputs/eval/confusion_matrix.png`) — discuta quais
  classes o modelo confunde entre si e uma hipótese do porquê.

## 5. Análise de erros
_(parte do critério 4)_

- 3-5 exemplos comentados de falsos positivos (de `outputs/eval/error_examples/`).
- 3-5 exemplos comentados de falsos negativos.
- Padrões observados: erros concentrados em quê? (oclusão, baixa luz,
  espécie visualmente parecida com outra classe, animal pequeno na imagem, etc.)

## 6. Aplicação em vídeo
_(critério 5 — 10%)_

- Descrição do vídeo de teste usado (fonte, duração).
- Prints do resultado de inferência (`outputs/video_inference/`).
- Se aplicável: rastreamento com ByteTrack (bônus) — descreva a diferença
  entre detecção frame-a-frame e rastreamento contínuo.

## 7. Limitações e próximos passos
_(parte do critério 6 — 10%)_

- Limitações de dataset (poucas imagens de espécies brasileiras reais,
  máscaras anotadas em subconjunto, ausência de imagens noturnas/chuva).
- O que mudaria com mais tempo/recursos (mais dados, mais classes, testar
  modelo maior, testar em câmera de borda real).

## Referências
- CBEE/UFLA — http://cbee.ufla.br
- Datasets citados individualmente conforme `data/DATASET_SOURCES.md`
- Bibliografia técnica (YOLO, Ultralytics, etc.)
