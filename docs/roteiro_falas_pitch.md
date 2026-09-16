# Roteiro do vídeo-pitch — demo ao vivo, sem slides

**Versão enxuta.** O roteiro anterior passava de 18 min na prática. Aqui as falas foram encurtadas e o
número de comandos caiu de 24 para **13**.

| | Fala | Gravação real (fala + digitar + esperar) |
|---|---|---|
| Roteiro completo | ≈ 7 min | **≈ 11 min** |
| Cortando tudo que está marcado com ✂️ | ≈ 5 min 30 | **≈ 8 min** |

Os passos com ✂️ são opcionais: corte-os se o ensaio passar do tempo. Cronometre um ensaio antes de
gravar — só assim você sabe o seu ritmo.

## Como ler

Cada passo tem sempre a mesma ordem:

1. ⌨️ **DIGITE:** cole o comando e aperte **Enter**.
2. 👀 **NA TELA:** o que deve aparecer. Espere aparecer antes de falar.
3. 🗣️ **FALE:** o texto, enquanto aquilo está na tela.

**Não fale e digite ao mesmo tempo**, com uma exceção marcada com ⏳ (o `yolo val`, que demora 13 s).
Depois de abrir uma imagem ou vídeo, feche a janela com `⌘ W` antes do próximo passo.

Os números vêm das execuções reais em `outputs/` — **não mude para valores "mais bonitos"**. Troque
"Integrante 1…5" pelos nomes. Para entender cada assunto a fundo: `docs/estudo_visao_computacional.md`.

---

## Preparação (antes de apertar "gravar")

1. Terminal com fonte grande (`⌘ +` até ~20 pt).
2. Cole este bloco **uma vez**:

```bash
cd ~/Documents/visao
source .venv/bin/activate
export IMG=data/processed/segment/images/test/inat_343605085.jpg
ver() { awk -v a=$2 -v b=$3 'NR>=a && NR<=b {printf "%4d  %s\n", NR, $0}' "$1"; }
yolo predict model=outputs/segment/weights/best.pt source=$IMG project=$PWD/outputs name=aquecimento exist_ok=True > /dev/null 2>&1
clear
```

   Ele ativa o ambiente, guarda a foto do tamanduá em `$IMG`, cria o atalho `ver arquivo início fim`
   (mostra linhas de código) e "aquece" o modelo, para a primeira execução gravada não ficar lenta.

3. Ensaio: `PAUSA=0 ABRIR=0 bash scripts/demo_video.sh` → tem que terminar com `Demonstração concluída.`
4. Assista `outputs/video_inference/video_teste.mp4` e anote **um** momento de erro (plantas marcadas
   como tamanduá **ou** a capivara nadando). As 9 capivaras aparecem em ≈ 0:19.

---

## Bloco 1 — Abertura (≈ 0:20) · Integrante 1

⌨️ **DIGITE:** nada. Câmera com o grupo.

🗣️ **FALE:**

> Olá! Somos o grupo _[nome]_, da disciplina de Visão Computacional, com o professor Romes Heriberto.
> Segundo o CBEE, da UFLA, **centenas de milhões de animais silvestres morrem atropelados todo ano nas
> rodovias brasileiras**. Nossa pergunta: uma câmera na beira da estrada consegue reconhecer esses
> animais sozinha e mostrar exatamente onde eles estão? Em vez de slides, vamos mostrar o sistema
> rodando.

---

## Bloco 2 — Proposta e pipeline (≈ 0:40) · Integrante 1

### Passo 2.1

⌨️ **DIGITE:**

```bash
ls scripts/ && ver config/classes.yaml 11 16
```

👀 **NA TELA:** a lista de scripts `00` a `05` e, embaixo, as 5 classes.

🗣️ **FALE:**

> Um estudo do ICMC da USP já usou YOLO para **detectar** mamíferos em rodovias, ou seja, desenhar uma
> caixa em volta do animal. Fomos além: fizemos **segmentação de instâncias**, que recorta o contorno
> exato de cada animal, e testamos em vídeo com rastreamento.
>
> O projeto é esta sequência de scripts: o **00** baixa as fotos, o **00b** gera as anotações, o **01**
> divide o dataset, o **02** e o **03** treinam, o **04** avalia e o **05** roda no vídeo. As classes são
> **capivara, cachorro-do-mato, tamanduá-bandeira, tatu e lobo-guará** — espécies muito atropeladas, e as
> duas últimas ameaçadas de extinção.

---

## Bloco 3 — Dataset (≈ 1:10) · Integrante 2

### Passo 3.1 — Coleta e anotação automática

⌨️ **DIGITE:**

```bash
clear && ver scripts/00b_autolabel_yoloe.py 39 48
```

👀 **NA TELA:** `TARGET_PROMPTS` ("capybara", "giant anteater"…) e `DISTRACTOR_PROMPTS`.

🗣️ **FALE:**

> Não existia dataset público com essas espécies. Montamos o nosso com o **iNaturalist**, usando só fotos
> com espécie confirmada pela comunidade e licença Creative Commons: **900 fotos, 180 por espécie**.
>
> Anotar centenas de contornos à mão não cabia no prazo, então usamos o **YOLOE-26**, um modelo
> *open-vocabulary*: em vez de classes fixas, ele recebe um texto, como "giant anteater", e procura
> aquilo na imagem. Embaixo estão os **distratores** — pássaro, pessoa, carro — para que esses objetos
> não virem o animal. A espécie vem do iNaturalist; o YOLOE só diz **onde** o animal está. São
> **pseudo-labels**, anotação automática, e essa é a nossa principal limitação.

### Passo 3.2 — Formato da anotação ✂️

⌨️ **DIGITE:**

```bash
clear && cat data/processed/detect/labels/test/inat_343605085.txt
```

👀 **NA TELA:** `2 0.465787 0.511431 0.373963 0.139824`

🗣️ **FALE:**

> Uma anotação é isto: o **2** é a classe, tamanduá, e os quatro números são o centro, a largura e a
> altura da caixa, em fração da imagem. Na segmentação, a linha traz os pontos do contorno.

### Passo 3.3 — EDA e divisão dos dados

⌨️ **DIGITE:**

```bash
clear && for s in train val test; do echo "$s: $(ls data/processed/detect/images/$s | wc -l | tr -d ' ') imagens"; done
open docs/figuras/eda_class_distribution.png
```

👀 **NA TELA:** `train: 499`, `val: 106`, `test: 108` e o gráfico de barras por classe.

🗣️ **FALE:**

> Descartando as fotos sem animal encontrado, ficaram **713 imagens com 835 animais**. O gráfico mostra o
> desbalanceamento: **218 capivaras contra 121 lobos-guará**. Dividimos em **70% treino, 15% validação e
> 15% teste**, com semente fixa: o modelo aprende no treino, a validação escolhe a melhor época, e o
> **teste só é usado no final**.

---

## Bloco 4 — O modelo e a demo (≈ 1:50) · Integrante 3

### Passo 4.1 — O que é o YOLO

⌨️ **DIGITE:**

```bash
clear && ver scripts/02_train_detect.py 35 51
```

👀 **NA TELA:** `model.train(` com `epochs`, `imgsz`, `batch`, `patience`, `mosaic`.

🗣️ **FALE:**

> **YOLO** quer dizer *You Only Look Once*. Detectores antigos trabalhavam em duas etapas: propunham
> regiões e depois classificavam cada uma. O YOLO faz tudo **numa única passada pela rede neural** — a
> imagem entra e sai a lista de caixas com classe e confiança. Por isso é rápido o bastante para vídeo.
> Por dentro é uma **rede convolucional**: as primeiras camadas aprendem bordas e texturas, as últimas,
> o animal inteiro.
>
> Usamos o **YOLO26n**, da Ultralytics; o **n** é de *nano*, 2,5 milhões de parâmetros, pensado para
> dispositivos pequenos como uma câmera de estrada. Não treinamos do zero: partimos de pesos treinados no
> **COCO** e ensinamos só as nossas cinco espécies — isso é ***transfer learning***.
>
> Nos hiperparâmetros: imagens de **640 pixels**, **batch 16**, **até 100 épocas**, ***early stopping***
> com paciência 20 e *augmentation* — o mosaico junta quatro fotos numa só, mais variação de cor e
> espelhamento, para a rede decorar menos.

### Passo 4.2 — Detecção ao vivo

⌨️ **DIGITE:**

```bash
clear && yolo predict model=outputs/detect/weights/best.pt source=$IMG project=$PWD/outputs name=demo_detect exist_ok=True
open outputs/demo_detect/inat_343605085.jpg
```

👀 **NA TELA:** `1 tamandua_bandeira, 7.1ms` e a foto noturna com a **caixa**.

🗣️ **FALE:**

> Esta é uma foto do conjunto de teste, que o modelo nunca viu. Em **7 milissegundos** ele achou o
> tamanduá-bandeira e desenhou a caixa com a confiança.

### Passo 4.3 — Segmentação ao vivo

⌨️ **DIGITE:**

```bash
clear && yolo predict model=outputs/segment/weights/best.pt source=$IMG project=$PWD/outputs name=demo_seg exist_ok=True
open outputs/demo_seg/inat_343605085.jpg
```

👀 **NA TELA:** a mesma foto, agora com o corpo do animal pintado.

🗣️ **FALE:**

> Agora o **YOLO26n-seg**: o mesmo detector com uma saída a mais, a **máscara**, que marca exatamente os
> pixels do animal.

### Passo 4.4 — O que o modelo devolve ✂️

⌨️ **DIGITE** (cole o bloco inteiro):

```bash
clear && python -c "
from ultralytics import YOLO
r = YOLO('outputs/segment/weights/best.pt')('$IMG', verbose=False)[0]
print('classe   :', r.names[int(r.boxes.cls[0])])
print('confiança:', round(float(r.boxes.conf[0]), 2))
print('caixa px :', [round(v) for v in r.boxes.xyxy[0].tolist()])
print('máscara  :', tuple(r.masks.data.shape), '| pontos do contorno:', len(r.masks.xy[0]))
"
```

👀 **NA TELA:** `tamandua_bandeira`, `0.77`, `[290, 460, 572, 585]`, `(1, 640, 640) | pontos: 157`.

🗣️ **FALE:**

> Por dentro, é isto que sai: a classe, a confiança de 0,77, a caixa em pixels e a máscara — uma matriz
> de 640 por 640 em que cada pixel diz se é animal ou não.

### Passo 4.5 — Caixa × máscara

⌨️ **DIGITE:**

```bash
open docs/figuras/box_vs_mask_tamandua.jpg
```

👀 **NA TELA:** caixa à esquerda, máscara à direita.

🗣️ **FALE:**

> Lado a lado fica claro o ganho: a caixa inclui estrada e mato; a máscara pega **só o corpo**. No teste,
> a máscara ocupa pouco mais da metade da área da caixa — quase metade de cada caixa é fundo. Numa
> rodovia, isso decide se o animal está **realmente sobre a pista**.

---

## Bloco 5 — Resultados e erros (≈ 1:40) · Integrante 4

### Passo 5.1 — Medindo ao vivo ⏳

⌨️ **DIGITE** (cole as três linhas juntas):

```bash
clear && echo "classe | imagens | instâncias | CAIXA: P  R  mAP50  mAP50-95 | MÁSCARA: P  R  mAP50  mAP50-95"
yolo val model=outputs/segment/weights/best.pt data=config/classes_seg.yaml split=test plots=False \
    project=$PWD/outputs name=demo_val exist_ok=True 2>&1 | grep -E "^ +(all|capivara|cachorro_do_mato|tamandua_bandeira|tatu|lobo_guara) "
```

⏳ **FALE ENQUANTO RODA** (13 s com a tela parada, é normal):

> Vamos medir o modelo **ao vivo** nas **104 imagens de teste, com 117 animais**. Dois conceitos:
> **precisão** é, do que o modelo marcou, quanto estava certo; **recall** é, dos animais que existiam,
> quantos ele achou. E **IoU** é a sobreposição entre a caixa prevista e a correta: conta acerto a partir
> de 50%.

👀 **NA TELA:** a linha `all  104  117  0.91  0.846  0.878  0.742  0.91  0.846  0.875  0.712`.

🗣️ **FALE:**

> O **mAP** junta precisão e recall numa média entre as classes. Nas **máscaras**, as quatro últimas
> colunas: **mAP@0.5 de 0,875**, precisão 0,91 e recall 0,85. O detector de caixas ficou em 0,85. O
> **mAP@0.5:0.95**, que exige sobreposição de 50% até 95%, cai para 0,71 nas máscaras — acertar o
> contorno exato é mais difícil que acertar a caixa. Lobo-guará e tatu são as classes mais fáceis;
> capivara e tamanduá, as mais difíceis.

### Passo 5.2 — Matriz de confusão

⌨️ **DIGITE:**

```bash
open docs/figuras/segment_confusion_matrix_normalized.png
```

👀 **NA TELA:** o quadrado azul, mais escuro na diagonal.

🗣️ **FALE:**

> Cada coluna é a espécie verdadeira, cada linha é o que o modelo previu, e a diagonal são os acertos:
> **lobo-guará 96%, tatu 95%, capivara e tamanduá 77%**. O modelo quase não troca uma espécie por outra;
> o erro comum é a linha *background*, ou seja, **não detectar o animal**.

### Passo 5.3 — Um erro nosso

⌨️ **DIGITE:**

```bash
open docs/figuras/fn1_tamandua_armadilha.jpg
```

👀 **NA TELA:** armadilha fotográfica, com as caixas verdes marcadas "FN".

🗣️ **FALE:**

> Aqui o modelo achou o tamanduá com **93% de confiança** e caixa justa. Mas a anotação automática, em
> verde, estava errada: uma das caixas marca um tronco. Conta como erro, só que quem errou foi a
> **label**, não o modelo. O ruído das pseudo-labels atrapalha o treino e a própria medição — é a nossa
> limitação central.

### Passo 5.4 — Um falso positivo ✂️

⌨️ **DIGITE:**

```bash
open docs/figuras/fp1_cachorro_noite.jpg
```

🗣️ **FALE:**

> Outro caso: cachorro-do-mato à noite, imagem borrada, dois animais lado a lado e a anotação marcou só
> um. Pouca luz, animais sobrepostos e label incompleta.

---

## Bloco 6 — Vídeo com rastreamento (≈ 1:20) · Integrante 5

### Passo 6.1 — O código do tracking

⌨️ **DIGITE:**

```bash
clear && ver scripts/05_infer_video.py 52 59
```

👀 **NA TELA:** `stream=True` e `model.track(tracker="bytetrack.yaml", ...)`.

🗣️ **FALE:**

> Um vídeo é uma sequência de imagens, aqui a 24 quadros por segundo, e o modelo roda em cada quadro. Mas
> o detector não tem memória: ele não sabe que a capivara de agora é a mesma de um instante atrás. O
> **ByteTrack** liga as detecções entre quadros — prevê para onde cada animal vai e casa a previsão com
> as caixas novas pela sobreposição. Cada animal ganha um **número de identificação**, o que permite
> contar indivíduos e estimar trajetória.

### Passo 6.2 — Capivaras

⌨️ **DIGITE:**

```bash
open outputs/video_inference/video_teste.mp4
```

👀 **NA TELA:** QuickTime; **espaço** para dar play, pulando o início preto.

🗣️ **FALE** (no começo):

> Vídeo real de **89 segundos**, com capivaras em Miranda, no Mato Grosso do Sul. Cada uma tem máscara e
> número próprios.

🗣️ **FALE** (em ≈ 0:19, com as 9 capivaras):

> Aqui são **nove capivaras ao mesmo tempo**. São 11 milissegundos por quadro: roda em tempo real.

🗣️ **FALE** (no momento de erro que você anotou):

> Ele também erra: _[plantas aquáticas marcadas como tamanduá / esta capivara nadando não é detectada]_.

### Passo 6.3 — Lobo-guará: a falha mais séria

⌨️ **DIGITE:**

```bash
open outputs/video_lobo_guara/video_lobo_guara.mp4
```

👀 **NA TELA:** lobo-guará atrás de grades, quase sempre rotulado "capivara".

🗣️ **FALE:**

> Num zoológico de Tóquio, a máscara segue bem o corpo, mas **o modelo chama o lobo-guará de capivara em
> 35% dos quadros, e de lobo-guará em só 7%** — sendo que era a melhor classe no teste com fotos. Isso é
> **mudança de domínio**: grades na frente, animal de costas, cenário que não existe no treino.
> Resultado bom no teste **não garante** resultado bom no mundo real.

---

## Bloco 7 — Fechamento (≈ 0:35) · Todos

⌨️ **DIGITE:**

```bash
clear && pytest -q
```

👀 **NA TELA:** `59 passed`. Depois, câmera com o grupo ou o repositório no GitHub.

🗣️ **FALE:**

> **Integrante 1:** Entregamos um pipeline completo e reproduzível, com 59 testes automáticos passando:
> coleta, anotação, detecção, segmentação, avaliação e vídeo com rastreamento.
>
> **Integrante 2:** As limitações são as labels automáticas, o dataset pequeno e desbalanceado e o erro
> de espécie quando o cenário muda.
>
> **Integrante 3:** Os próximos passos são revisar as labels à mão e incluir imagens noturnas, de
> armadilha fotográfica e de rodovia, além de outros animais como exemplos negativos.
>
> **Integrante 4:** Também queremos testar modelos maiores e rodar numa câmera de borda, com alerta em
> tempo real.
>
> **Integrante 5:** Código, notebook e instruções estão no GitHub, no link na tela. **Obrigado!**

---

## Se ainda passar do tempo

1. Corte os passos ✂️ (3.2, 4.4 e 5.4).
2. No bloco 6, mostre **só** o vídeo das capivaras e resuma o lobo-guará em uma frase.
3. No bloco 4, corte a explicação dos hiperparâmetros (fica só o "o que é YOLO").
4. Corte trechos parados na edição: a espera do `yolo val`, a digitação e a abertura das janelas.

## Números para ter na ponta da língua (teste)

| Modelo | Saída | Precisão | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| YOLO26n (detecção) | caixas | 0,849 | 0,807 | 0,847 | 0,744 |
| YOLO26n-seg | máscaras | 0,910 | 0,846 | 0,875 | 0,712 |

- 900 fotos baixadas → 713 com label (835 animais) → 499 / 106 / 108 imagens. Teste efetivo: 104 imagens,
  117 animais (4 GIFs ignorados).
- Erros (conf ≥ 0,25, IoU ≥ 0,5): detecção 95 acertos, 22 FP, 22 FN; segmentação 101 acertos, 28 FP, 16 FN.
- Velocidade (MPS, 640 px): ~7 ms de inferência por imagem.

## Checklist antes de gravar

- [ ] Ensaio cronometrado, dentro do limite.
- [ ] Bloco de preparação colado no terminal que vai ser gravado.
- [ ] Momento de erro do vídeo das capivaras anotado.
- [ ] Notebook Colab com **saídas visíveis** (Item 3 da entrega).
- [ ] Repositório no GitHub criado, para mostrar no fechamento.
- [ ] Nomes do grupo e revisão manual das labels preenchidos (README, seção 6).
- [ ] Créditos na descrição: iNaturalist (CC), Wikimedia Commons (Martin Thurnherr, CC BY-SA 4.0; Tokyo
      Zoo, CC BY 3.0), YOLOE-26 e Ultralytics.
- [ ] Uso de IA generativa declarado no relatório.
- [ ] YouTube como **não listado**, link testado numa aba anônima.
