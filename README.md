# Sistema de Visão Computacional para Detecção e Segmentação de Fauna Silvestre em Rodovias

**Disciplina:** Visão Computacional e Reconhecimento de Padrões — Pós-graduação
**Cenário:** Cidades Inteligentes (variante: prevenção de atropelamento de fauna silvestre)
**Prof.:** Romes Heriberto

## 1. Problema

Rodovias brasileiras registram um número extremamente alto de atropelamentos de fauna
silvestre todos os anos, com estimativas na casa das centenas de milhões de animais por
ano segundo o Centro Brasileiro de Estudos em Ecologia de Estradas (CBEE/UFLA). Este
projeto constrói um sistema de detecção + segmentação de instâncias capaz de identificar
espécies silvestres brasileiras em imagens/vídeo, como base para um futuro sistema de
alerta em tempo real para motoristas (câmeras de borda em rodovias).

Trabalho relacionado: pesquisa do ICMC-USP (São Carlos) testou modelos YOLO para
detecção de mamíferos da fauna brasileira em rodovias (publicado em *Scientific
Reports*), mas sem etapa de segmentação — esse é o diferencial deste projeto.

## 2. Classes-alvo

| Classe | Nome científico | Justificativa |
|---|---|---|
| Capivara | *Hydrochoerus hydrochaeris* | Uma das espécies mais frequentemente atropeladas; grande porte, alta visibilidade |
| Cachorro-do-mato | *Cerdocyon thous* | Espécie mais atropelada em estudos de mamíferos no Brasil central |
| Tamanduá-bandeira | *Myrmecophaga tridactyla* | Espécie ameaçada de extinção, forma corporal muito distinta (bom caso de segmentação) |
| Tatu | *Euphractus sexcinctus* (tatu-peba) | Entre as espécies mais atropeladas; carapaça com contorno bem definido |
| Lobo-guará | *Chrysocyon brachyurus* | Espécie ameaçada, ícone de conservação da fauna do Cerrado |

## 3. Estrutura do repositório

```
wildlife-roadkill-cv/
├── config/
│   └── classes.yaml          # definição das classes e data.yaml do YOLO
├── data/
│   ├── raw/                  # dataset baixado (não versionado)
│   ├── processed/            # splits train/val/test em formato YOLO
│   └── DATASET_SOURCES.md    # fontes públicas + plano de anotação própria
├── scripts/
│   ├── 01_prepare_dataset.py
│   ├── 02_train_detect.py
│   ├── 03_train_seg.py
│   ├── 04_evaluate.py
│   └── 05_infer_video.py
├── tests/                    # pytest das funções puras de 01 e 04
├── notebooks/
│   └── main_colab.ipynb      # notebook executável (Item 3 da entrega)
├── docs/
│   ├── fase1_proposta.md
│   ├── relatorio_tecnico_template.md
│   ├── roteiro_video_pitch.md
│   └── AGENT_PROMPT.md
└── outputs/                   # pesos, métricas, vídeos gerados (gitignored)
```

## 4. Como reproduzir

Todos os comandos devem ser rodados **a partir da raiz do repositório** — os
caminhos em `config/*.yaml` são relativos a ela.

```bash
git clone <URL_DO_REPO>
cd wildlife-roadkill-cv
pip install -r requirements.txt

# 0. Coletar fotos reais (iNaturalist, licenças CC, atribuição em data/ATRIBUICAO_imagens.csv)
python scripts/00_download_inaturalist.py --out_dir data/inaturalist --per_class 180
#    pré-anotar caixas + máscaras com YOLOE-26 (pseudo-labels; revise os mosaicos em outputs/autolabel/)
mkdir -p models && curl -L -o models/yoloe-26l-seg.pt \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt
python scripts/00b_autolabel_yoloe.py --model models/yoloe-26l-seg.pt --overwrite

# 1. Preparar dataset (gerado pelo passo 0, ou um export YOLO do Roboflow em data/raw/;
#    aceita raw/images + raw/labels ou raw/{train,valid,test}/{images,labels})
python scripts/01_prepare_dataset.py --raw_dir data/raw --out_dir data/processed
#    subconjunto com máscaras (polygon) para a segmentação:
python scripts/01_prepare_dataset.py --raw_dir data/raw_seg --out_dir data/processed --task segment

# 2. Treinar detector
python scripts/02_train_detect.py --data config/classes.yaml --epochs 100 --imgsz 640

# 3. Treinar segmentação
python scripts/03_train_seg.py --data config/classes_seg.yaml --epochs 100 --imgsz 640

# 4. Avaliar no conjunto de teste (mAP, IoU médio por classe, exemplos de FP/FN)
python scripts/04_evaluate.py --weights outputs/detect/weights/best.pt --data config/classes.yaml --split test
#    segmentação + figuras caixa x máscara:
python scripts/04_evaluate.py --weights outputs/segment/weights/best.pt --data config/classes_seg.yaml \
    --split test --name eval_seg --compare_boxes_masks

# 5. Rodar inferência em vídeo (com tracking opcional)
python scripts/05_infer_video.py --weights outputs/segment/weights/best.pt --source data/video_teste.mp4 --track
```

Os scripts 02 a 05 aceitam `--device` (`0` para GPU CUDA, `mps` para Mac Apple
Silicon, `cpu`); sem o argumento o Ultralytics escolhe sozinho.

Ou abra `notebooks/main_colab.ipynb` no Google Colab — ele clona este repositório
e executa as mesmas etapas com GPU.

### Ambiente local e testes

```bash
python3.12 -m venv .venv          # ou: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                             # testes das funções puras (sem GPU nem dataset)
```

No Mac com Apple Silicon use um Python **arm64** (`python -c "import platform; print(platform.machine())"`
deve imprimir `arm64`); um Python x86_64 (Homebrew em `/usr/local`) roda sob Rosetta e não tem wheels
recentes do PyTorch.

**IDE (IntelliJ IDEA Ultimate 2024.2 ou PyCharm):** o IntelliJ precisa dos plugins
*Python* e *Python Community Edition* **da mesma versão da IDE** (242.x) — um
*Python Community Edition* de versão mais nova impede o plugin Python de carregar.
Para notebooks, instale também *Jupyter* e *Notebooks Core*. Depois abra a pasta do
repositório, vá em *File → Project Structure → SDKs → + → Python SDK → Existing
environment* e aponte para `.venv/bin/python`. Os testes rodam pelo ícone ▶ ao lado
de cada função em `tests/`; ao criar configurações de execução dos scripts, mantenha
o *working directory* na raiz do repositório.

## 5. Ferramentas

Google Colab (GPU) · PyTorch/Ultralytics YOLO26 · OpenCV · CVAT/Roboflow (anotação) ·
supervision (visualização/tracking) · GitHub.

## 6. Integridade acadêmica

Uso de IA generativa (Claude / Claude Code) como apoio na estruturação do repositório,
templates de documentação, revisão de código, escrita dos scripts de coleta e
pré-anotação e execução do pipeline — declarado conforme exigido no edital.

**Origem das labels:** as caixas e máscaras do dataset são **pseudo-labels geradas
automaticamente** pelo modelo open-vocabulary YOLOE-26 (`scripts/00b_autolabel_yoloe.py`);
a classe de cada imagem vem da identificação *research grade* do iNaturalist, não do
modelo. Não há dados sintéticos: todas as imagens são fotos reais com licença CC
(`data/ATRIBUICAO_imagens.csv`). Revisão manual das labels pelo grupo: _[descrever
quantas imagens foram revisadas/corrigidas]_. A análise dos resultados e a redação
do relatório são de responsabilidade do grupo.
