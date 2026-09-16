# Estudo estruturado — Visão Computacional para entender (e defender) este projeto

Material de estudo para quem vai apresentar o projeto **sem ter feito tudo sozinho**. Ele parte do
zero ("o que é um pixel") e chega aos detalhes do nosso código e dos nossos números.

**Como usar:**

1. Leia na ordem. Cada parte depende da anterior.
2. No fim de cada seção há um **"Veja no projeto"** com comandos para rodar. **Rode.** Ver o modelo
   funcionando fixa muito mais do que só ler.
3. A **Parte VI** tem as perguntas prováveis do professor com respostas curtas. Treine respondendo em voz
   alta sem olhar.

Antes de rodar qualquer comando deste documento, abra o terminal e cole:

```bash
cd ~/Documents/visao && source .venv/bin/activate
export IMG=data/processed/segment/images/test/inat_343605085.jpg
```

## Sumário

- **Parte I — Fundamentos:** 1. Imagem digital · 2. Tarefas de visão computacional · 3. Redes neurais
  convolucionais · 4. Como uma rede é treinada
- **Parte II — Detecção e segmentação:** 5. Detecção de objetos e YOLO · 6. Segmentação de instâncias ·
  7. Modelos open-vocabulary e pseudo-labels
- **Parte III — Avaliação:** 8. IoU, precisão, recall, mAP, matriz de confusão · 9. Análise de erros ·
  10. Splits, vazamento e reprodutibilidade
- **Parte IV — Vídeo e mundo real:** 11. Vídeo e rastreamento (ByteTrack) · 12. Generalização, domínio
  e limitações
- **Parte V — O projeto de ponta a ponta:** 13. Mapa do pipeline · 14. Script por script · 15.
  Ferramentas
- **Parte VI — Treino para a apresentação:** 16. Perguntas prováveis · 17. Glossário · 18. Plano de
  estudo e checklist

---

# Parte I — Fundamentos

## 1. O que é uma imagem digital

- Uma imagem é uma **matriz de números**. Cada posição é um **pixel**.
- Uma imagem colorida tem **3 canais** (vermelho, verde e azul, o **RGB**). Cada pixel tem 3 valores de
  0 a 255. Uma foto de 1024×1024 pixels é, portanto, um bloco de números de 1024 × 1024 × 3.
  - O **OpenCV** (biblioteca que usamos) guarda os canais na ordem **BGR**, não RGB. É uma pegadinha
    clássica.
- **HSV** é outra forma de representar cor: **H**ue (matiz, "qual cor"), **S**aturation (saturação,
  "quão viva") e **V**alue (brilho). Usamos HSV na EDA do relatório: V médio < 60 indica foto escura
  (noturna), e S médio < 20 indica foto quase sem cor (armadilha fotográfica infravermelha).
- **Resolução:** as fotos do iNaturalist têm em média 923 × 766 px. A rede não recebe a imagem nesse
  tamanho; ela é redimensionada para **640 px** (seção 5.4).
- **Normalização:** antes de entrar na rede, os valores 0–255 viram 0–1. Redes treinam melhor com
  números pequenos.

**Veja no projeto:**

```bash
python -c "import cv2; im = cv2.imread('$IMG'); print(im.shape, im.dtype, im[0,0])"
# (1024, 1024, 3) uint8 [b g r]  -> altura, largura, canais; valores 0–255
```

## 2. As tarefas de visão computacional

```
Classificação          Detecção                 Segmentação semântica     Segmentação de instâncias
"tem um tamanduá"      caixa + classe           cada pixel tem classe     cada pixel tem classe E indivíduo
                       ┌──────────┐             ░░░░░░░░░░                ░░░░ ▓▓▓▓
   [tamanduá]          │ tamanduá │             ░ animal ░                ░#1░ ▓#2▓   (duas capivaras
                       │   0,77   │             ░░░░░░░░░░                ░░░░ ▓▓▓▓    separadas)
                       └──────────┘
```

| Tarefa | Pergunta que responde | Saída | No projeto |
|---|---|---|---|
| Classificação | *O que* tem na imagem? | 1 rótulo | não usamos sozinha |
| **Detecção** | O que tem e *onde* (aproximadamente)? | caixas + classe + confiança | `02_train_detect.py` (YOLO26n) |
| Segmentação semântica | Qual classe tem cada pixel? | 1 mapa de classes (não separa indivíduos) | não usamos |
| **Segmentação de instâncias** | Onde está *cada indivíduo*, pixel a pixel? | caixa + classe + **máscara por objeto** | `03_train_seg.py` (YOLO26n-seg) |
| **Rastreamento (tracking)** | Esse objeto é o mesmo do quadro anterior? | ID persistente ao longo do vídeo | `05_infer_video.py --track` (ByteTrack) |

**Por que segmentação de instâncias é o diferencial:** a caixa diz "o animal está mais ou menos aqui". A
máscara diz "*este* conjunto de pixels é *este* animal". Isso permite medir a área real ocupada, a
silhueta, a postura e a direção, separar animais encostados (capivaras em grupo) e cruzar a máscara com
a região da pista.

## 3. Redes neurais convolucionais (CNN)

### 3.1 Neurônio, camada, rede

- Uma **rede neural** é uma função enorme com milhões de números ajustáveis, os **pesos** ou
  **parâmetros**. O YOLO26n tem 2,5 milhões; o YOLO26n-seg, 3,05 milhões.
- **Treinar** é ajustar esses pesos para que a saída fique próxima da resposta correta (seção 4).

### 3.2 Convolução

- Um **filtro** (kernel) é uma pequena matriz, por exemplo 3×3, que **desliza** pela imagem. Em cada
  posição, multiplica seus valores pelos pixels embaixo e soma. O resultado é um **mapa de
  características** (*feature map*).
- Um filtro pode "acender" onde há uma borda vertical, outro onde há uma textura de pelo.
- **A rede aprende sozinha os valores dos filtros.** Ninguém programa "detector de focinho".
- **Stride** (passo) e **pooling** reduzem o tamanho do mapa. Cada camada vê uma região maior da imagem
  original (o **campo receptivo** cresce).

### 3.3 Hierarquia de características

```
imagem → [bordas, cores] → [texturas, cantos] → [partes: focinho, orelha, carapaça] → [animal inteiro]
         camadas iniciais    intermediárias        profundas                           cabeça (saída)
```

### 3.4 Anatomia de um detector moderno

| Parte | Função | Analogia |
|---|---|---|
| **Backbone** | extrai características da imagem (a CNN "que enxerga") | os olhos |
| **Neck** (FPN/PAN) | combina mapas de várias escalas: detalhes finos (animais pequenos) com contexto (animais grandes) | juntar o zoom com a visão geral |
| **Head** | a partir das características, prevê caixa, classe e (na seg) coeficientes de máscara | a boca que responde |

- **GFLOPs** (bilhões de operações de ponto flutuante) medem o custo de uma imagem passar pela rede.
  YOLO26n: 5,9 GFLOPs; YOLO26n-seg: 10,3. Menos FLOPs = mais rápido e menor consumo, o que importa numa
  câmera de borda.

**Veja no projeto:**

```bash
python -c "from ultralytics import YOLO; YOLO('outputs/segment/weights/best.pt').info()"
# YOLO26n-seg summary: 309 layers, 3,054,875 parameters, ..., 10.3 GFLOPs
```

## 4. Como uma rede é treinada

### 4.1 O ciclo

```
imagem de treino → rede → predição → compara com a label (LOSS) → calcula gradientes → ajusta pesos
      └───────────────────────────── repete para cada batch, por várias épocas ─────────────────────┘
```

- **Loss (função de perda):** um número que mede o quanto a predição está errada. Treinar é minimizar a
  loss. No `outputs/detect/results.csv` aparecem:
  - `box_loss`: erro de **localização** da caixa (baseado em IoU).
  - `cls_loss`: erro de **classe**.
  - `l1_loss`: erro absoluto nas coordenadas da caixa (termo do modo end-to-end do YOLO26).
  - Na segmentação, há também `seg_loss`: erro da **máscara**.
- **Gradiente descendente:** a derivada da loss indica em que direção mexer cada peso para o erro
  diminuir. O **otimizador** faz esse ajuste. No nosso caso, com `optimizer=auto`, o Ultralytics escolheu
  **AdamW** (ele só escolhe MuSGD em treinos com mais de 10 mil iterações).
- **Learning rate (taxa de aprendizado):** o tamanho do passo. Grande demais, o treino "pula" a solução;
  pequeno demais, fica lento. Usamos **warmup**: começa pequeno nas 3 primeiras épocas e depois sobe.

### 4.2 Vocabulário do treino

| Termo | Significado | Nosso valor |
|---|---|---|
| **Época** | uma passada completa por todas as imagens de treino | até 100 |
| **Batch** | quantas imagens a rede vê antes de cada ajuste dos pesos | 16 |
| **imgsz** | tamanho da imagem de entrada | 640 px |
| **Patience** | quantas épocas sem melhora na validação antes de parar (early stopping) | 20 |
| **Seed** | semente aleatória fixa, para reproduzir o resultado | 42 |
| **best.pt / last.pt** | pesos da melhor época na validação / da última época | usamos `best.pt` |

### 4.3 Overfitting (sobreajuste) e early stopping

- **Overfitting:** a rede **decora** o treino em vez de aprender o padrão. A loss de treino continua
  caindo, mas a de validação para de cair ou sobe.
- No nosso detector, a `val/box_loss` sobe levemente no final e o mAP de validação estabiliza por volta
  da época 75. É um sinal de início de sobreajuste, esperado com só 499 imagens de treino.
- **Early stopping** interrompe o treino quando a validação não melhora por `patience` épocas. A
  segmentação parou na época 86 e a melhor foi a 66.
- Como o `best.pt` é escolhido: pela **fitness** na validação, uma combinação ponderada das métricas
  dominada pelo mAP@0.5:0.95. Na segmentação ela considera caixa e máscara juntas; por isso a "melhor
  época" (66) não é exatamente a época de maior mAP de máscara.

### 4.4 Data augmentation (aumento de dados)

Criar variações artificiais das imagens de treino para a rede generalizar melhor:

| Técnica | O que faz | Por que ajuda |
|---|---|---|
| **Mosaico** | junta 4 imagens numa só, em quadrantes | mais animais e contextos por imagem, objetos em escalas diferentes |
| **Flip horizontal** (0,5) | espelha metade das imagens | um animal virado para a esquerda continua sendo o mesmo animal |
| **HSV** (h 0,015, s 0,7, v 0,4) | altera matiz, saturação e brilho | robustez a luz diferente (dia, entardecer, câmera ruim) |
| Translação, escala, RandAugment, random erasing | desloca, aproxima/afasta, apaga pedaços | robustez a posição, tamanho e oclusão parcial |

- **close_mosaic = 10:** o mosaico é **desligado nas 10 últimas épocas**, para a rede terminar treinando
  com imagens parecidas com as reais. É por isso que a `box_loss` cai de repente na época 90.

### 4.5 Transfer learning / fine-tuning

- Treinar do zero exigiria centenas de milhares de imagens. Em vez disso, partimos de pesos
  **pré-treinados no COCO** (~118 mil imagens de treino, 80 classes: pessoa, carro, cachorro, cavalo…).
- As camadas iniciais já sabem "enxergar" bordas, texturas e formas de animais. Só **reajustamos** a rede
  e **refazemos a cabeça de classificação** para as nossas 5 classes: 606 de 708 tensores foram
  aproveitados.
- **Por que funciona com só ~500 imagens:** a parte difícil (aprender a ver) já estava pronta.

**Veja no projeto:**

```bash
open outputs/detect/train_batch0.jpg        # um batch de treino: veja o MOSAICO e as cores alteradas
open outputs/detect/results.png             # curvas de loss e métricas por época
ver() { awk -v a=$2 -v b=$3 'NR>=a && NR<=b {printf "%4d  %s\n", NR, $0}' "$1"; }
ver scripts/02_train_detect.py 33 52        # a chamada model.train() com os hiperparâmetros
grep -E "^(optimizer|lr0|close_mosaic|mosaic|patience|imgsz|batch):" outputs/detect/args.yaml
```

---

# Parte II — Detecção e segmentação

## 5. Detecção de objetos e o YOLO

### 5.1 O que um detector devolve

Para cada objeto encontrado: **caixa** (bounding box), **classe** e **confiança** (0 a 1, o quanto o
modelo "acredita" na detecção).

Formatos de caixa que aparecem no código:

| Formato | Significado | Onde aparece |
|---|---|---|
| `xyxy` | canto superior esquerdo (x1, y1) e inferior direito (x2, y2), em pixels | `r.boxes.xyxy`, `compute_iou` |
| `xywh` | centro (x, y), largura, altura | `r.boxes.xywh` |
| `xywhn` | igual, mas **normalizado** (0–1, fração da imagem) | labels YOLO, `r.boxes.xywhn` |

> Origem (0,0) é o **canto superior esquerdo**; y cresce para **baixo**.

### 5.2 Formato das labels YOLO

Um `.txt` por imagem, **uma linha por objeto**:

```
Detecção:     classe  cx  cy  w  h                    (5 valores)
              2 0.465787 0.511431 0.373963 0.139824   → tamanduá; centro em 46,6% × 51,1%;
                                                        caixa com 37,4% da largura e 14,0% da altura
Segmentação:  classe  x1 y1  x2 y2  x3 y3 ...          (polígono do contorno, também normalizado)
```

Normalizar (0–1) faz a label valer para qualquer tamanho da imagem. O `01_prepare_dataset.py` distingue
os dois formatos pela quantidade de valores na linha (`parse_label_file`).

### 5.3 One-stage × two-stage

| | Two-stage (ex.: Faster R-CNN, Mask R-CNN) | One-stage (YOLO) |
|---|---|---|
| Como funciona | 1) propõe regiões candidatas; 2) classifica e ajusta cada uma | uma única passada prevê todas as caixas e classes |
| Velocidade | mais lento | **rápido, tempo real** |
| Precisão histórica | era maior | hoje equivalente na maioria dos casos |
| Uso típico | análise offline | vídeo, câmeras, dispositivos de borda (**nosso caso**) |

### 5.4 Como o YOLO funciona, passo a passo

1. **Pré-processamento (letterbox):** redimensiona a imagem mantendo a proporção e completa com bordas
   até 640 px. No vídeo 16:9, o log mostra `384x640`: menos borda, múltiplo de 32.
2. **Backbone** extrai características em 3 escalas (mapas de 80×80, 40×40 e 20×20 para uma entrada de
   640). As escalas finas pegam animais pequenos, e as grossas pegam animais grandes.
3. **Neck** mistura as escalas.
4. **Head:** cada posição de cada mapa prevê "há um objeto com centro aqui?", a caixa e a pontuação de
   cada classe. É **anchor-free**: prevê a caixa diretamente, sem caixas-modelo pré-definidas.
5. **Pós-processamento:**
   - descarta predições com confiança abaixo do limiar (`conf`, padrão 0,25);
   - **NMS (Non-Maximum Suppression)** nos YOLOs anteriores: várias posições vizinhas acham o mesmo
     animal; o NMS mantém a caixa de maior confiança e apaga as que se sobrepõem muito a ela (IoU alto).
6. Converte as caixas de volta para o tamanho original da foto.

### 5.5 A família YOLO e o YOLO26

- **YOLO v1 (2016, Redmon et al.)** introduziu a ideia "You Only Look Once". Seguiram v2, v3, v4 e v5,
  e depois as versões da Ultralytics: **YOLOv8** (2023, anchor-free), **YOLO11** (2024) e **YOLO26**
  (2025–2026).
- **O que o YOLO26 muda** (conferido nos arquivos do pacote instalado, `cfg/models/26/yolo26.yaml`):
  - `end2end: True`: **inferência end-to-end, sem NMS**. A cabeça aprende a dar **uma única** predição
    por objeto, então não precisa do passo de NMS. Isso é mais simples e mais previsível em
    dispositivos de borda. (É por isso que o comentário do `00b_autolabel_yoloe.py` diz que o YOLOE-26
    "não usa NMS".)
  - `reg_max: 1`: **remove o DFL** (Distribution Focal Loss), uma forma mais complexa de regredir a caixa
    usada no v8/11. Por isso o `results.csv` não tem `dfl_loss`.
  - Traz o otimizador **MuSGD** (não foi usado aqui, ver 4.1).
- **Tamanhos:** `n` (nano), `s`, `m`, `l`, `x`. Quanto maior, mais preciso e mais lento. Escolhemos o
  **n** por ser compatível com câmera de borda e por treinar rápido num notebook.

### 5.6 Limiar de confiança (conf): um trade-off

- `conf` **alto** (ex.: 0,7): menos alarmes falsos (↑ precisão), mas perde animais (↓ recall).
- `conf` **baixo** (ex.: 0,1): acha quase tudo (↑ recall), com mais lixo (↓ precisão).
- Num sistema de alerta de atropelamento, **perder um animal** (falso negativo) costuma ser pior que um
  alarme falso. Isso justificaria um conf mais baixo.

**Veja no projeto:**

```bash
yolo predict model=outputs/detect/weights/best.pt source=$IMG project=$PWD/outputs name=conf90 conf=0.9 exist_ok=True
yolo predict model=outputs/detect/weights/best.pt source=$IMG project=$PWD/outputs name=conf05 conf=0.05 exist_ok=True
open outputs/conf90/inat_343605085.jpg outputs/conf05/inat_343605085.jpg
# Com conf 0.9 o tamanduá (0,72) SOME. Com 0.05 podem aparecer caixas extras de baixa confiança.
cat data/processed/detect/labels/test/inat_343605085.txt
```

## 6. Segmentação de instâncias (YOLO-seg)

### 6.1 Máscara

- Uma **máscara** é uma matriz do tamanho da imagem com 0 (fundo) ou 1 (objeto). Existe **uma máscara
  por instância**.
- Na nossa demo: `r.masks.data.shape = (1, 640, 640)`, ou seja, 1 animal e máscara de 640×640.
- A mesma máscara pode ser representada como **polígono** (a lista de pontos do contorno), que é o
  formato da label: `r.masks.xy` tem 157 pontos para o tamanduá.

### 6.2 Como o YOLO-seg gera máscaras (ideia do YOLACT)

```
             ┌─► head de detecção ─► caixa + classe + k COEFICIENTES por animal
backbone+neck┤
             └─► módulo "proto"   ─► k MÁSCARAS-PROTÓTIPO para a imagem inteira (baixa resolução)

máscara do animal = sigmoide( coeficientes · protótipos )  → recorta dentro da caixa → amplia → limiar 0,5
```

- Os **protótipos** são como "pincéis" genéricos (contornos, regiões de primeiro plano). Cada animal
  combina esses pincéis com seus próprios pesos.
- **Vantagem:** a máscara sai quase de graça, a partir das mesmas características da detecção. Por isso
  o YOLO26n-seg custa só um pouco mais que o detector (7,2 ms × 5,6 ms de inferência no teste).
- Alternativas: **Mask R-CNN** (two-stage, mais lento) e **SAM** (Segment Anything, segmenta qualquer
  coisa a partir de um clique ou caixa, mas não classifica espécies e é pesado).

### 6.3 O que a segmentação acrescentou, em números

- Nas labels do teste, a máscara ocupa em média **55% da área da caixa**; nas predições, **58%**.
  **Quase metade de cada caixa é fundo.**
- Para decidir "o animal está sobre a pista?", cruzar **máscara × região da pista** é quase 2× mais
  preciso em área do que cruzar a caixa.

**Veja no projeto:**

```bash
open outputs/eval_seg/box_vs_mask/*.jpg      # 6 exemplos: caixa (esquerda) × máscara (direita)
ver scripts/04_evaluate.py 211 226           # compare_boxes_masks: r.plot(masks=False) × r.plot(boxes=False)
head -c 300 data/processed/segment/labels/test/inat_343605085.txt; echo   # polígono da label
```

## 7. Modelos open-vocabulary e pseudo-labels

### 7.1 Conjunto fechado × vocabulário aberto

- **Conjunto fechado (closed-set):** o nosso YOLO26 só conhece as 5 classes do treino. Tudo o que vê,
  ele tenta encaixar numa delas. É por isso que urubus viram "capivara" (FP4).
- **Open-vocabulary:** o **YOLOE** recebe **texto** ("giant anteater") e procura aquilo.
  - Um **encoder de texto** (MobileCLIP, arquivo `models/mobileclip2_b.ts`) transforma cada prompt num
    vetor de números (*embedding*).
  - A rede compara esse vetor com as características de cada região da imagem. Onde forem parecidos,
    há detecção.
  - `model.set_classes([...])` define os prompts (`00b_autolabel_yoloe.py`, linha 192).
- Essa ideia vem do **CLIP** (OpenAI, 2021), treinado com milhões de pares imagem-texto para colocar
  "foto de tamanduá" e o texto "anteater" perto no mesmo espaço de vetores.

### 7.2 Como geramos as labels (`00b_autolabel_yoloe.py`)

1. A **espécie já é conhecida** (iNaturalist, *research grade*). O YOLOE só precisa **localizar**.
2. Prompts da espécie + genéricos ("mammal", "animal") + **distratores** ("bird", "person", "car"…).
   Se o YOLOE achar um carro, a detecção sai com o nome "car" e é descartada, em vez de virar "tatu".
3. Filtros: confiança ≥ 0,3 e área ≥ 0,2% da imagem (`select_instances`).
4. **Deduplicação:** o mesmo animal pode ser achado por "capybara" e por "mammal". Mantemos só a de
   maior confiança se o IoU entre elas for ≥ 0,7 (`dedupe_instances`).
5. Grava a caixa (detecção) e o polígono (segmentação), com a **classe do iNaturalist**.
6. Fotos sem nenhuma detecção válida são descartadas (187 de 900).

### 7.3 Pseudo-labels: o que são e qual o risco

- **Pseudo-label:** anotação gerada por um modelo, não por um humano.
- É uma forma de **destilação**: um modelo grande (YOLOE-26**l**) "ensina" um pequeno (YOLO26**n**).
- **Riscos que apareceram:** caixa deslocada, indivíduo faltando e animal+filhote fundidos (FP1, FN1,
  FN3).
- **Consequência dupla:** (1) o modelo **treina** com erros; (2) o **teste também tem erros**, então as
  métricas medem a **concordância com o YOLOE**, não com a verdade. Às vezes o modelo acerta e é
  penalizado (FN1).

**Veja no projeto:**

```bash
ver scripts/00b_autolabel_yoloe.py 39 50      # prompts e distratores
ver scripts/00b_autolabel_yoloe.py 62 92      # select_instances e dedupe_instances
open outputs/autolabel/revisao_capivara.jpg   # mosaico de revisão
python -c "import json; r = json.load(open('outputs/autolabel/report.json')); [print(k, v['rotuladas'], '/', v['imagens'], 'instâncias:', v['instancias']) for k, v in r['classes'].items()]"
```

---

# Parte III — Avaliação

## 8. Métricas: IoU, precisão, recall, mAP e matriz de confusão

### 8.1 IoU (Intersection over Union)

```
IoU = área da interseção / área da união
    = inter / (área_A + área_B − inter)
```

- 0 = não se tocam; 1 = idênticas. É o critério de "acertou o lugar?".
- **Exemplo real (nossa demo):** a caixa prevista pela segmentação é `[290, 460, 572, 585]`; a label
  convertida para pixels é `[285,5; 452,1; 668,4; 595,3]`.
  - interseção = 282 × 125 = 35.250 px² (a predição está inteira dentro da label)
  - área da label = 382,9 × 143,2 ≈ 54.830 px²
  - **IoU ≈ 35.250 / 54.830 ≈ 0,64** → ≥ 0,5 → **acerto**, embora a label seja mais comprida (inclui a
    cauda).
- **Código:** `scripts/04_evaluate.py`, função `compute_iou` (linhas 28–38).

### 8.2 TP, FP e FN

Com limiar de confiança (0,25) e de IoU (0,5), cada predição e cada label vira:

| | Definição | Exemplo |
|---|---|---|
| **TP** (verdadeiro positivo) | predição com **classe certa** e **IoU ≥ 0,5** com uma label ainda não usada | tamanduá achado no lugar certo |
| **FP** (falso positivo) | predição sem label correspondente: lugar errado, **classe errada** ou caixa duplicada | urubu marcado como capivara |
| **FN** (falso negativo) | label que nenhuma predição encontrou | capivara nadando não detectada |

> Não existe "verdadeiro negativo" em detecção: não dá para contar "todos os lugares sem animal".

**Casamento guloso** (`match_predictions`, linha 66): ordena as predições pela confiança e cada uma
pega a label de mesma classe com maior IoU ainda livre.

### 8.3 Precisão, recall e F1

```
Precisão = TP / (TP + FP)   → "do que eu marquei, quanto estava certo?"
Recall   = TP / (TP + FN)   → "dos animais que existiam, quantos eu achei?"
F1       = 2·P·R / (P + R)  → média harmônica, um número só
```

- **Nossa análise de erros (conf 0,25):** detecção com 95 TP, 22 FP e 22 FN → P = 95/117 = 0,81 e R =
  95/117 = 0,81.
- **Por que o Ultralytics mostra P = 0,849 e não 0,81?** O `model.val()` não usa conf fixo: ele varre
  todos os limiares e reporta P e R **no limiar que maximiza o F1** (veja `outputs/eval/BoxF1_curve.png`).
  São duas medições legítimas com critérios diferentes. **Saiba explicar isso.**

### 8.4 Curva precisão × recall, AP e mAP

1. Ordena todas as predições de uma classe pela confiança.
2. Descendo o limiar, calcula P e R a cada passo → **curva PR**.
3. **AP (Average Precision)** = área sob a curva PR (o Ultralytics interpola em 101 pontos, padrão COCO).
   Vai de 0 a 1.
4. **mAP** = média do AP entre as classes.

| Métrica | Critério de acerto | Leitura |
|---|---|---|
| **mAP@0.5** (mAP50) | IoU ≥ 0,5 | "achou o animal no lugar aproximadamente certo?" |
| **mAP@0.5:0.95** (mAP50-95) | média de 10 limiares: 0,50, 0,55, …, 0,95 | **mais rigorosa**: exige caixa/máscara precisa |
| **mAP de máscara** | mesmo cálculo, mas o IoU é entre **máscaras** (pixels), não caixas | mede a qualidade do contorno |

**Por que o mAP@0.5:0.95 das máscaras (0,712) é menor que o das caixas (0,742)?** Nos limiares altos
(0,85–0,95), acertar o contorno pixel a pixel é muito mais difícil que acertar 4 coordenadas.

### 8.5 IoU médio por classe (nossa métrica extra)

- É calculado **só nos acertos** (TP). Por isso é alto (0,89–0,95): mede *quão justas são as caixas
  corretas*. Os erros aparecem no recall e nas contagens de FP e FN, não aqui.

### 8.6 Matriz de confusão

- **Colunas = classe verdadeira**; **linhas = classe prevista**. Há uma linha e uma coluna extras,
  **background**:
  - linha *background* = animal verdadeiro que o modelo **não detectou** (FN);
  - coluna *background* = detecção **onde não havia animal** (FP).
- **Normalizada por coluna:** cada coluna soma 1, então a diagonal é "a fração de cada espécie
  corretamente detectada".
- **Nossa leitura (segmentação):** diagonal com lobo-guará 0,96, tatu 0,95, cachorro 0,84, capivara 0,77
  e tamanduá 0,77. **Pouca confusão entre espécies; o erro dominante é não detectar.** "Capivara" é o
  rótulo mais usado nos FPs de fundo, porque o modelo encaixa objetos marrons ou mamíferos na classe
  mais frequente.

### 8.7 Cuidado com conclusões em teste pequeno

- 117 instâncias → **cada animal vale ≈ 0,9 ponto percentual de recall**. Diferenças de 2–3 pontos
  entre detecção e segmentação **não são conclusivas**.

**Veja no projeto:**

```bash
yolo val model=outputs/detect/weights/best.pt data=config/classes.yaml split=test plots=False \
    project=$PWD/outputs name=estudo_val exist_ok=True 2>&1 | grep -E "^ +(all|capivara|cachorro_do_mato|tamandua_bandeira|tatu|lobo_guara) "
open outputs/eval/BoxPR_curve.png outputs/eval/BoxF1_curve.png outputs/eval/confusion_matrix_normalized.png
python -c "
import importlib.util as u
s = u.spec_from_file_location('ev', 'scripts/04_evaluate.py'); ev = u.module_from_spec(s); s.loader.exec_module(ev)
print('IoU da demo:', round(ev.compute_iou([290, 460, 572, 585], [285.5, 452.1, 668.4, 595.3]), 3))
print('IoU sem sobreposição:', ev.compute_iou([0, 0, 10, 10], [20, 20, 30, 30]))
print('IoU metade deslocada:', round(ev.compute_iou([0, 0, 10, 10], [5, 0, 15, 10]), 3))
"
open outputs/eval/val_batch0_labels.jpg outputs/eval/val_batch0_pred.jpg   # label × predição lado a lado
```

## 9. Análise de erros: pensar como cientista

Métrica diz **quanto** erra; a análise de erros diz **por quê**. Para cada exemplo: **o que aconteceu →
padrão → hipótese → o que faria para corrigir**.

| Exemplo (`docs/figuras/`) | O que aconteceu | Padrão | Correção possível |
|---|---|---|---|
| `fp1_cachorro_noite.jpg` | 2 animais, label com 1; caixas sobre a dupla | baixa luz + sobreposição + label incompleta | revisar labels; mais fotos noturnas |
| `fp2_lobo_tela_camera.jpg` | foto da tela de uma câmera; previu tatu 0,39 e cachorro 0,32 | fora da distribuição | conf mais alto elimina |
| `fp3_tatu_closeup.jpg` | 2 caixas no mesmo tatu | enquadramento parcial | mais close-ups no treino |
| `fp4_urubus_capivara.jpg` | urubus segmentados como capivara (0,92) | **conjunto fechado** | imagens negativas / classe "outro animal" |
| `fn1_tamandua_armadilha.jpg` | modelo acertou (0,93), label errada (tronco) | **ruído de pseudo-label** | revisão manual |
| `fn2_capivaras_rio.jpg` | 3 capivaras em fila, 1 não detectada | oclusão + submersa | mais fotos de grupos |
| `fn3_tamandua_filhote.jpg` | mãe + filhote fundidos na label | ambiguidade de anotação | regra de anotação clara |

## 10. Splits, vazamento de dados e reprodutibilidade

- **Treino** (499 imagens): a rede ajusta os pesos. **Validação** (106): escolhe a melhor época e dispara
  o early stopping. **Teste** (108; efetivo 104): **usado uma única vez**, no final.
- Por que 3 conjuntos? Se escolhêssemos a época pelo teste, o teste deixaria de ser "nunca visto" e o
  resultado ficaria otimista.
- **Vazamento de dados (data leakage):** a mesma informação cair em treino e teste. Cuidados no código:
  - **uma foto por observação** do iNaturalist (fotos da mesma observação são quase idênticas);
  - `01_prepare_dataset.py` **apaga os splits anteriores** antes de gerar novos (senão, com outra seed, a
    mesma imagem poderia ficar em treino e teste);
  - o **mesmo split** é usado para detecção e segmentação (teste idêntico nas duas tarefas).
- **Reprodutibilidade:** `seed=42` na divisão (`random.Random(seed).shuffle`) e no treino
  (`deterministic=True`).
- **Os 15 GIFs:** arquivos GIF com extensão `.jpg`. O Ultralytics lê o formato real e os ignora; o
  `04_evaluate.py` aplica o mesmo filtro (`split_by_real_format`) para os totais da análise de erros
  baterem com o mAP.
- **Desbalanceamento:** capivara 218 × lobo-guará 121 instâncias (1,8×). A classe mais frequente tende a
  "puxar" as predições incertas (visto nos FPs e no vídeo do lobo-guará).

**Veja no projeto:**

```bash
ver scripts/01_prepare_dataset.py 79 92
for s in train val test; do echo "$s: $(ls data/processed/detect/images/$s | wc -l | tr -d ' ')"; done
cat data/processed/eda/eda_summary_detect.json
```

---

# Parte IV — Vídeo e mundo real

## 11. Vídeo e rastreamento (ByteTrack)

### 11.1 Vídeo = sequência de imagens

- **FPS** (quadros por segundo): o vídeo das capivaras tem ~24 fps e 2.147 quadros em 89 s.
- **Tempo real** = processar cada quadro em menos de 1/fps segundos (≈ 42 ms a 24 fps). Nosso modelo gasta
  ≈ 1,4 + 6,6 + 3,1 ≈ **11 ms por quadro** (pré-processamento + inferência + pós-processamento, no MPS).
- `stream=True` (`05_infer_video.py`, linha 54): processa um quadro por vez com um **gerador**, sem
  guardar milhares de máscaras na memória.

### 11.2 Detecção quadro a quadro × rastreamento

- **Sem tracking:** cada quadro é independente. O modelo não sabe que a capivara do quadro 100 é a do
  quadro 99. Não dá para contar indivíduos nem medir trajetória.
- **Com tracking (tracking-by-detection):** o detector acha as caixas; o **tracker** liga as caixas entre
  quadros e dá um **ID** a cada animal.

### 11.3 Como o ByteTrack funciona

```
quadro t-1: tracks ativos (ID 1, 2, 3...)
      │  1. FILTRO DE KALMAN prevê onde cada track estará no quadro t (posição + velocidade)
      ▼
quadro t: detecções do YOLO
      │  2. 1ª associação: detecções de ALTA confiança (≥ 0,25) × previsões, por IoU (algoritmo húngaro)
      │  3. 2ª associação: detecções de BAIXA confiança (0,1–0,25) × tracks que sobraram
      │     (a grande ideia do ByteTrack: um animal meio encoberto tem confiança baixa, mas ainda é ele)
      │  4. detecção de alta confiança sem par → NOVO ID
      │  5. track sem detecção → "perdido", guardado por 30 quadros (track_buffer) antes de apagar
      ▼
quadro t: tracks atualizados
```

- **Filtro de Kalman:** modelo matemático que estima posição e velocidade a partir de medições ruidosas.
  Prevê "a caixa deve estar aqui agora" e corrige com a detecção real.
- **Algoritmo húngaro:** encontra o melhor pareamento global entre previsões e detecções (maximizando o
  IoU total).
- **Parâmetros usados** (`ultralytics/cfg/trackers/bytetrack.yaml`): `track_high_thresh 0.25`,
  `track_low_thresh 0.1`, `new_track_thresh 0.25`, `track_buffer 30`, `match_thresh 0.8`.
- **Detalhe fino (ótimo para mostrar maturidade):** rodamos o vídeo com `conf=0.3`, e esse filtro
  acontece **antes** do tracker. Assim, as detecções de baixa confiança (0,1–0,3) nunca chegam à 2ª
  associação do ByteTrack. Rodar o tracking com `conf` menor poderia recuperar animais parcialmente
  encobertos, como a capivara nadando.
- **ID switch:** quando dois animais se cruzam e trocam de ID. É o erro típico de tracking.
- **Resultado:** até **9 capivaras** rastreadas no mesmo quadro (quadro 441, ≈ 0:19).

**Veja no projeto:**

```bash
ver scripts/05_infer_video.py 43 59
grep -v "^#" .venv/lib/python3.12/site-packages/ultralytics/cfg/trackers/bytetrack.yaml | grep -v "^$"
grep "frame 441/" outputs/logs/05_video_teste.log      # o quadro com 9 capivaras
yolo track model=outputs/segment/weights/best.pt source=data/video_teste.mp4 show=True save=False   # Ctrl C para parar
```

## 12. Generalização, mudança de domínio e limitações

- **Generalização:** funcionar em dados diferentes dos de treino. É o objetivo real.
- **Mudança de domínio (domain shift):** a distribuição dos dados reais é diferente da do treino.
  - **Treino:** fotos de naturalistas, de perto, bem enquadradas, animal de lado.
  - **Vídeo do lobo-guará:** zoológico, grades cortando a silhueta, animal de costas.
  - **Resultado:** "capivara" em 35% dos quadros e "lobo-guará" em só 7,1%, **mesmo o lobo-guará tendo
    mAP@0.5 = 0,985 no teste**. **Métrica boa no teste ≠ funcionar no mundo real.**
- **Problema de conjunto fechado:** sem exemplos negativos, qualquer coisa vira uma das 5 classes (urubus,
  plantas aquáticas → tamanduá).
- **Câmera de rodovia real:** distância grande (animal pequeno), ângulo de cima, noite, chuva, faróis e
  desfoque de movimento. Nada disso está no dataset.
- **Próximos passos coerentes:** revisar labels; fotos noturnas, de armadilha e de grupos; **imagens
  negativas** (estrada vazia, outros animais); vídeos reais de rodovia para validar e ajustar;
  modelos maiores (s/m); limiar por classe; exportar para **ONNX/TensorRT/CoreML** e medir numa placa
  embarcada.

---

# Parte V — O projeto de ponta a ponta

## 13. Mapa do pipeline

```
 iNaturalist (API)                                     YOLOE-26l-seg + prompts de texto
       │                                                         │
       ▼                                                         ▼
 00_download_inaturalist.py ─► data/inaturalist/<espécie>/ ─► 00b_autolabel_yoloe.py
   900 fotos CC + ATRIBUICAO.csv                               │
                                         ┌─────────────────────┴───────────────────┐
                                         ▼                                         ▼
                              data/raw (caixas)                         data/raw_seg (polígonos)
                                         │ 01_prepare_dataset.py (EDA + split 70/15/15, seed 42)
                                         ▼                                         ▼
                        data/processed/detect/{images,labels}/{train,val,test}   .../segment/...
                                         │                                         │
                    02_train_detect.py (YOLO26n, COCO→5 classes)      03_train_seg.py (YOLO26n-seg)
                                         ▼                                         ▼
                        outputs/detect/weights/best.pt              outputs/segment/weights/best.pt
                                         │                                         │
                                         └──────────► 04_evaluate.py ◄─────────────┤
                                            mAP, P, R, matriz, IoU/classe,         │
                                            FP/FN, caixa×máscara                   ▼
                                                                          05_infer_video.py --track
                                                                         vídeo com máscaras + IDs
```

## 14. Script por script

### `00_download_inaturalist.py` — coleta

- **Entrada → saída:** API do iNaturalist → `data/inaturalist/<classe>/inat_<id>.jpg` +
  `data/ATRIBUICAO_imagens.csv`.
- **Decisões:** `quality_grade=research` (espécie confirmada); só licenças CC0/BY/BY-SA/BY-NC; anotação
  "Evidence of Presence = Organism" (sem fezes nem pegadas); **1 foto por observação** (evita vazamento);
  miniatura trocada por `large` (até 1024 px); pausa de 1 s entre requisições (limite da API); CSV
  regravado a cada classe (não perde nada se cair).
- **Conceitos:** coleta de dados, licenças, vazamento.

### `00b_autolabel_yoloe.py` — pseudo-labels

- **Entrada → saída:** fotos + CSV → `data/raw` (caixas), `data/raw_seg` (polígonos),
  `outputs/autolabel/report.json` + mosaicos.
- **Funções-chave:** `prompts_for`, `select_instances` (conf/área), `box_iou` + `dedupe_instances`,
  `format_bbox_line` / `format_polygon_line` (formato YOLO; polígono < 3 pontos é inválido),
  `review_mosaic`.
- **Conceitos:** open-vocabulary, embeddings de texto, pseudo-labels, deduplicação por IoU.

### `01_prepare_dataset.py` — EDA e split

- **Entrada → saída:** `data/raw(_seg)` → `data/processed/<task>/{images,labels}/{train,val,test}` +
  `data/processed/eda/`.
- **Funções-chave:** `label_path_for` (troca `images`→`labels`, mesma regra do Ultralytics),
  `split_dataset` (determinístico), `parse_label_file` (bbox × polígono), `check_raw_class_order` (evita
  ids trocados), `run_eda`.
- **Conceitos:** EDA, desbalanceamento, splits, reprodutibilidade.

### `02_train_detect.py` e `03_train_seg.py` — treino

- **Entrada → saída:** `config/classes(_seg).yaml` → `outputs/detect|segment/` (pesos, `results.csv`,
  curvas, matriz de validação).
- **Detalhes:** checkpoint base `yolo26n.pt` / `yolo26n-seg.pt`; `project` com caminho absoluto (no
  Ultralytics 8.4+ um caminho relativo iria para `~/runs`); `--device mps` no Mac.
- **Conceitos:** transfer learning, hiperparâmetros, augmentation, early stopping, overfitting.

### `04_evaluate.py` — avaliação

- **Entrada → saída:** pesos + yaml + split → `outputs/eval(_seg)/` (métricas JSON, matriz, curvas,
  `error_examples/`, `box_vs_mask/`).
- **Funções-chave:** `compute_iou`, `load_gt_boxes` (polígono vira caixa envolvente), `match_predictions`
  (TP/FP/FN), `mean_iou_per_class`, `split_by_real_format` (filtro dos GIFs), `analyze_errors` (desenha FN
  em verde), `compare_boxes_masks`.
- **Conceitos:** IoU, TP/FP/FN, P/R, mAP, matriz de confusão, análise de erros.

### `05_infer_video.py` — vídeo

- **Entrada → saída:** vídeo (ou webcam `0`, ou `rtsp://`) → vídeo anotado em `outputs/video_inference/`.
- **Detalhes:** valida o arquivo antes de carregar o modelo; `stream=True`; `model.track(tracker=
  "bytetrack.yaml")` com `--track`.
- **Conceitos:** FPS, tempo real, tracking-by-detection, Kalman, ByteTrack.

### `tests/` — testes automáticos

- `pytest` roda **59 testes** das funções puras (IoU, casamento TP/FP/FN, split, parsing de labels,
  seleção e deduplicação do autolabel, download). **Não precisa de GPU nem do dataset.**
- Por que importa: garante que a **matemática das métricas** está correta. Se o `compute_iou` estivesse
  errado, todos os números da análise de erros estariam errados.

```bash
pytest -q
grep -n "def test" tests/test_evaluate.py
```

## 15. Ferramentas

| Ferramenta | Papel |
|---|---|
| **Python 3.12** | linguagem |
| **PyTorch 2.14** | framework de redes neurais (tensores, gradientes, GPU) |
| **MPS** (Metal Performance Shaders) | backend do PyTorch que usa a GPU do Mac Apple Silicon (M3 Pro); `device=mps` |
| **CUDA** | equivalente para GPUs NVIDIA (Colab); `device=0` |
| **Ultralytics 8.4.150** | biblioteca do YOLO26/YOLOE: `train`, `val`, `predict`, `track`, CLI `yolo` |
| **OpenCV** (`cv2`) | leitura, desenho e escrita de imagens e vídeo (BGR!) |
| **CLIP / MobileCLIP** | encoder de texto do YOLOE |
| **lap** | resolve o pareamento (algoritmo húngaro) do ByteTrack |
| **matplotlib, pandas** | gráficos da EDA, leitura do `results.csv` |
| **pytest** | testes |
| **Jupyter / Colab / nbconvert** | notebook executável da entrega |

---

# Parte VI — Treino para a apresentação

## 16. Perguntas prováveis (responda em voz alta, depois confira)

1. **Por que YOLO e não Mask R-CNN?** YOLO é one-stage e roda em tempo real (≈ 7 ms por imagem). O
   cenário é uma câmera de rodovia, que precisa de velocidade e hardware pequeno. Mask R-CNN é two-stage
   e bem mais lento.
2. **Por que o tamanho nano?** 2,5–3 M de parâmetros, compatível com dispositivo de borda, e treina rápido
   num notebook. Testar s/m é um próximo passo.
3. **O que é mAP@0.5:0.95 e por que é menor nas máscaras?** É a média do AP em 10 limiares de IoU, de 0,5 a
   0,95. Nos limiares altos, acertar o contorno pixel a pixel é mais difícil que acertar a caixa.
4. **Como garantem que o teste não vazou?** Uma foto por observação, split com seed fixa gerado uma vez,
   splits antigos apagados, teste usado só no final e melhor época escolhida pela validação.
5. **As métricas são confiáveis, se as labels são automáticas?** Parcialmente. Elas medem a concordância
   com o YOLOE. Há casos em que o modelo acerta e é penalizado (FN1). Por isso declaramos a limitação e o
   próximo passo é revisar à mão.
6. **Por que a capivara é difícil se é a classe mais numerosa?** Mais fotos em grupo, animais sobrepostos
   e submersos, e pseudo-labels menos consistentes nesses casos. Quantidade não compensa dificuldade.
7. **Diferença entre segmentação semântica e de instâncias?** A semântica diz "estes pixels são
   capivara". A de instâncias diz "estes são a capivara 1, aqueles a capivara 2".
8. **O que o ByteTrack faz que o detector não faz?** Dá memória entre quadros: Kalman prevê a posição,
   IoU + algoritmo húngaro casam com as detecções e cada animal ganha um ID. O diferencial é usar também
   detecções de baixa confiança.
9. **Por que o modelo errou o lobo-guará no vídeo?** Mudança de domínio (zoológico, grades, animal de
   costas) + conjunto fechado + desbalanceamento, que puxa para "capivara".
10. **Por que a precisão do Ultralytics (0,849) é diferente da contagem TP/FP (0,81)?** O Ultralytics
    reporta P e R no limiar de confiança que maximiza o F1; nossa contagem usa conf fixo de 0,25.
11. **O que é NMS e por que o YOLO26 não usa?** O NMS apaga caixas duplicadas do mesmo objeto. O YOLO26 é
    end-to-end: a cabeça é treinada para dar uma predição por objeto.
12. **Por que o IoU médio é ~0,9 se o mAP é ~0,85?** O IoU médio é calculado só nos acertos; os erros
    entram no mAP, no recall e nas contagens de FP e FN.
13. **O que é fine-tuning?** Partir de pesos pré-treinados (COCO) e continuar o treino no nosso dataset,
    trocando a cabeça para 5 classes.
14. **Para que serve o mosaico e por que desligar no fim?** Dá mais variedade de contexto e escala. Nas
    últimas 10 épocas é desligado para o modelo se ajustar a imagens reais.
15. **Houve overfitting?** Início, no fim do treino do detector: a `val/box_loss` subiu levemente enquanto a
    de treino caía. O early stopping e o `best.pt` mitigam.
16. **Como a máscara é gerada?** A cabeça dá coeficientes por animal; um módulo gera máscaras-protótipo;
    a combinação linear passa por sigmoide, é recortada pela caixa e passa por limiar.
17. **Como a label de segmentação é guardada?** Classe seguida dos pontos do polígono normalizados (0–1).
18. **Por que 640 px?** É o padrão de pré-treino e equilibra detalhe × velocidade. Animais muito pequenos
    ou distantes perdem detalhe; subir o imgsz é uma opção.
19. **Como virar um sistema real?** Validar com vídeos de rodovia, adicionar negativos, exportar para
    ONNX/TensorRT/CoreML, rodar numa placa embarcada e disparar alerta quando a máscara intersecta a pista
    por N quadros seguidos (tracking evita alarme por um quadro isolado).
20. **O que foi feito com IA generativa?** Declarado no README e no relatório: estrutura, scripts, revisão
    e execução com apoio do Claude Code. As decisões, a revisão e a análise são do grupo. _[Completar com o
    que o grupo fez.]_

## 17. Glossário rápido

| Termo | Definição curta |
|---|---|
| Anchor-free | prevê a caixa diretamente, sem caixas-modelo pré-definidas |
| AP / mAP | área sob a curva precisão×recall / média entre classes |
| Augmentation | variações artificiais das imagens de treino |
| Backbone / Neck / Head | extrator de características / mistura de escalas / camadas de saída |
| Batch | imagens processadas antes de cada ajuste de pesos |
| Bounding box | retângulo que envolve o objeto |
| ByteTrack | rastreador que associa detecções entre quadros usando também as de baixa confiança |
| CNN | rede neural convolucional |
| COCO | dataset de 80 classes usado no pré-treino |
| Confiança (conf) | pontuação 0–1 da detecção; limiar descarta as fracas |
| Conjunto fechado | modelo só conhece as classes do treino |
| Domain shift | dados reais diferentes dos de treino |
| Early stopping | parar quando a validação não melhora |
| Embedding | vetor numérico que representa texto ou imagem |
| End-to-end (YOLO26) | inferência sem NMS |
| Época | uma passada por todo o treino |
| F1 | média harmônica de precisão e recall |
| Feature map | saída de uma camada convolucional |
| Fine-tuning / transfer learning | reaproveitar pesos pré-treinados |
| FP / FN / TP | falso positivo / falso negativo / verdadeiro positivo |
| GFLOPs | custo computacional por imagem |
| ID switch | tracker troca o ID de dois objetos |
| Inferência | usar o modelo treinado para prever |
| IoU | interseção ÷ união de duas caixas ou máscaras |
| Filtro de Kalman | prevê posição e velocidade a partir de medições ruidosas |
| Letterbox | redimensionar mantendo proporção e completar com bordas |
| Loss | número que mede o erro; o treino minimiza |
| Máscara | matriz 0/1 que marca os pixels do objeto |
| MPS | GPU do Mac no PyTorch |
| NMS | remove caixas duplicadas do mesmo objeto |
| Open-vocabulary | detecta classes descritas por texto |
| Overfitting | decorar o treino e piorar na validação |
| Polígono | contorno da máscara como lista de pontos |
| Precisão / Recall | acertos ÷ marcados / acertos ÷ existentes |
| Pseudo-label | label gerada por modelo, não por humano |
| Segmentação de instâncias | máscara separada para cada objeto |
| Seed | semente aleatória para reprodutibilidade |
| Split | divisão treino/validação/teste |
| Tracking | seguir o mesmo objeto ao longo do vídeo |
| Vazamento (leakage) | informação do teste presente no treino |

## 18. Plano de estudo e checklist

**Sessão 1 (≈ 1h30) — fundamentos e YOLO:** seções 1–5. Rode os blocos "Veja no projeto" das seções
1, 3, 4 e 5 (inclusive o teste com `conf=0.9` e `conf=0.05`).

**Sessão 2 (≈ 1h30) — segmentação, métricas e erros:** seções 6–10. Calcule o IoU do exemplo à mão e
confira com o Python. Rode o `yolo val` e explique cada coluna em voz alta. Abra as 7 figuras de erro e
conte a história de cada uma.

**Sessão 3 (≈ 1h) — vídeo, projeto inteiro e ensaio:** seções 11–16. Rode o `yolo track ... show=True`.
Responda as 20 perguntas sem olhar. Faça o ensaio completo de `docs/roteiro_falas_pitch.md` cronometrando.

**Consigo explicar, sem ler, em 30 segundos cada:**

- [ ] O que é o YOLO e por que ele é rápido
- [ ] Detecção × segmentação de instâncias × tracking
- [ ] O que é transfer learning e por que funcionou com ~500 imagens
- [ ] O que são época, batch, early stopping e mosaico
- [ ] O formato de uma linha de label (caixa e polígono)
- [ ] O que é o YOLOE e o que é pseudo-label, com o risco
- [ ] IoU, com um exemplo numérico
- [ ] Precisão × recall, e qual importa mais num alerta de atropelamento
- [ ] mAP@0.5 × mAP@0.5:0.95
- [ ] Como ler a matriz de confusão (incluindo *background*)
- [ ] Um falso positivo e um falso negativo nossos, com hipótese
- [ ] Como o ByteTrack mantém o ID
- [ ] Por que o lobo-guará falhou no vídeo (mudança de domínio)
- [ ] Três limitações e três próximos passos
