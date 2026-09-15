# Roteiro de falas — Vídeo-pitch (≈ 7 min)

Baseado na estrutura de `roteiro_video_pitch.md`. Todos os números abaixo vêm das
execuções reais em `outputs/` (14/09/2026) — **não altere para valores "mais bonitos"**.
Troque "Integrante 1…5" pelos nomes; se o grupo for menor, redistribua os blocos
(o barema exige que **todos** apareçam/falem).

Texto falado: ~1.000 palavras ≈ 7 min em ritmo calmo. Leia em voz alta uma vez cronometrando.

---

## Bloco 1 — Abertura (0:00–0:40) · Integrante 1

**Na tela:** todos os integrantes (câmera) ou slide com título + nomes.

> Olá! Nós somos o grupo _[nome do grupo]_, da disciplina de Visão Computacional com o
> professor Romes Heriberto.
>
> Segundo o Centro Brasileiro de Estudos em Ecologia de Estradas, da UFLA, **centenas de
> milhões de animais silvestres morrem atropelados todo ano nas rodovias brasileiras**,
> muitos de espécies ameaçadas.
>
> Nossa pergunta foi: uma câmera na beira da estrada consegue reconhecer esses animais
> sozinha e mostrar exatamente onde eles estão?

## Bloco 2 — Contexto e proposta (0:40–1:20) · Integrante 1

**Na tela:** slide com as 5 espécies (fotos) e a frase "detecção + segmentação".

> Nosso cenário é Cidades Inteligentes, com foco em prevenção de atropelamento de fauna.
>
> Um estudo do ICMC da USP, publicado na *Scientific Reports*, já usou YOLO para
> **detectar** mamíferos brasileiros em rodovias, ou seja, desenhar uma caixa em volta do
> animal. Nós fomos além: nosso sistema faz **detecção e segmentação de instâncias**, que
> recorta o contorno exato de cada animal. E testamos tudo em vídeo real, com
> rastreamento.
>
> Escolhemos cinco espécies muito atropeladas: **capivara, cachorro-do-mato,
> tamanduá-bandeira, tatu-peba e lobo-guará**. O tamanduá e o lobo-guará são ameaçados de
> extinção.

## Bloco 3 — Dataset e EDA (1:20–2:20) · Integrante 2

**Na tela:** `data/processed/eda/eda_class_distribution.png` e depois um mosaico de
`outputs/autolabel/revisao_*.jpg`.

> Não havia dataset público anotado com essas espécies. Então montamos o nosso com o
> **iNaturalist**: só fotos com espécie confirmada pela comunidade e licença Creative
> Commons. Foram **900 fotos, 180 por espécie**, com a atribuição de cada uma no
> repositório.
>
> Para anotar, usamos o modelo **YOLOE-26**, que gerou as caixas e as máscaras
> automaticamente. A espécie veio do iNaturalist, não do modelo. Deixamos claro: são
> **pseudo-labels**, e não anotação manual.
> _[Se o grupo revisou labels: "Revisamos manualmente X imagens."]_
>
> Descartando as fotos em que o animal não foi encontrado, ficaram **713 imagens com 835
> animais**. O gráfico mostra o desbalanceamento: **218 capivaras contra 121
> lobos-guará**.
>
> Dividimos em **70% treino, 15% validação e 15% teste**, com semente fixa. O teste nunca
> foi visto no treino.

## Bloco 4 — Metodologia + demo ao vivo (2:20–3:40) · Integrante 3

**Na tela:** terminal rodando o `yolo predict` (detecção e depois segmentação) numa foto
de teste; depois `outputs/eval_seg/box_vs_mask/box_vs_mask_inat_343605085.jpg`
(tamanduá à noite).

> Usamos o **YOLO26** da Ultralytics: o **YOLO26n** para detecção e o **YOLO26n-seg**
> para segmentação, os dois com *fine-tuning* a partir de pesos do COCO.
>
> Os hiperparâmetros foram: imagem de **640 pixels, batch 16, até 100 épocas**, *early
> stopping* com paciência 20 e o *augmentation* padrão, com mosaico, espelhamento e
> variação de cor. O detector rodou as 100 épocas. A segmentação parou antes, e o melhor
> modelo foi o da **época 66**.
>
> _(rodando a demo)_ Aqui está uma foto de teste. O detector desenha a caixa com a
> espécie e a confiança. Agora o modelo de segmentação, na mesma foto.
>
> _(caixa × máscara)_ Lado a lado fica claro o que a segmentação acrescenta. A caixa deste
> tamanduá inclui estrada e mato, enquanto a máscara pega **só o corpo**. Assim sabemos
> a silhueta, a postura e a direção do animal, e dá para separar animais encostados. Numa
> rodovia, isso ajuda a saber se ele está realmente em cima da pista.

## Bloco 5 — Resultados e análise de erros (3:40–5:20) · Integrante 4

**Na tela:** tabela de métricas (slide), `outputs/eval_seg/confusion_matrix_normalized.png`,
depois `outputs/eval/error_examples/fp_00_inat_703929831_n2.jpg` e
`outputs/eval/error_examples/fn_00_inat_457895527_n2.jpg`.

Slide sugerido (conjunto de **teste**):

| Modelo | mAP@0.5 | mAP@0.5:0.95 | Precisão | Recall |
|---|---|---|---|---|
| Detecção (YOLO26n) — caixas | 0,847 | 0,744 | 0,849 | 0,807 |
| Segmentação (YOLO26n-seg) — máscaras | 0,875 | 0,712 | 0,910 | 0,846 |

> No **conjunto de teste**, o detector chegou a **mAP@0.5 de 0,85** e **mAP@0.5:0.95 de
> 0,74**. A segmentação teve **mAP@0.5 de 0,875 nas máscaras**, com precisão de 0,91 e
> recall de 0,85. Nos acertos, o IoU médio ficou acima de 0,88 em todas as classes.
>
> _(matriz de confusão)_ Na matriz de confusão, lobo-guará e tatu são as classes mais
> fáceis, com **96% e 95%** de acerto. Capivara e tamanduá ficam em **77%**. O modelo
> quase não confunde uma espécie com outra: o erro mais comum é **não detectar o
> animal**, o que acontece com 16% dos cachorros-do-mato e dos tamanduás.
>
> _(falso positivo)_ Dois erros. Este cachorro-do-mato à noite, com imagem borrada, está
> na verdade com **outro animal ao lado**, mas a anotação automática marcou só um. O
> modelo pegou os dois numa caixa só. Hipótese: pouca luz, animais sobrepostos e uma
> pseudo-label incompleta.
>
> _(falso negativo)_ Nesta armadilha fotográfica, o modelo achou o tamanduá com **93% de
> confiança**, com uma caixa justa. Mas a anotação automática tinha caixas erradas, uma
> delas num tronco de árvore. Aqui quem errou foi a **label**. Essa é nossa principal
> limitação: o ruído das pseudo-labels afeta o treino e a própria medição.

## Bloco 6 — Vídeo em ação (5:20–6:40) · Integrante 5

**Na tela:** `outputs/video_inference/video_teste.mp4` (capivaras), pulando o início
preto. Depois um trecho curto de `outputs/video_lobo_guara/video_lobo_guara.mp4`.

> Agora o vídeo real: **89 segundos** de capivaras em Miranda, no Mato Grosso do Sul,
> publicado no Wikimedia Commons.
>
> Cada capivara ganha uma máscara e um **número de identificação** que se mantém entre os
> quadros. É o rastreamento com **ByteTrack**, que chega a acompanhar **sete capivaras ao
> mesmo tempo**.
>
> Ele também erra: aqui, plantas aquáticas foram marcadas como tamanduá. E esta capivara
> nadando, só com a cabeça fora da água, não é detectada.
>
> _(vídeo do lobo-guará)_ No segundo vídeo, um lobo-guará num zoológico de Tóquio, aparece
> a limitação mais séria. A máscara segue bem o corpo, mas **o modelo chama o lobo-guará
> de capivara** na maior parte do tempo, mesmo tendo sido a melhor classe no teste com
> fotos. Nossa hipótese é **mudança de domínio**: grades na frente, animal de costas e um
> cenário que não aparece no treino. Resultado bom no teste não garante resultado bom no
> mundo real.

## Bloco 7 — Fechamento (6:40–7:20) · Todos

**Na tela:** todos os integrantes; slide com link do GitHub.

> **Integrante 1:** Entregamos um pipeline completo e reproduzível: coleta, anotação,
> detecção, segmentação, avaliação e vídeo com rastreamento.
>
> **Integrante 2:** As limitações são labels automáticas, um dataset pequeno e
> desbalanceado, e erro de espécie quando o cenário muda.
>
> **Integrante 3:** Os próximos passos são revisar as labels à mão e colocar mais imagens
> noturnas, de armadilha fotográfica e de rodovia.
>
> **Integrante 4:** Também queremos testar modelos maiores e rodar numa câmera de borda,
> com alerta em tempo real.
>
> **Integrante 5:** Código, notebook e instruções estão no GitHub, no link na tela.
>
> **Todos:** Obrigado!

---

## Checklist antes de gravar

- [x] 15 imagens GIF com extensão `.jpg` ignoradas (10 treino, 1 validação, 4 teste);
      avaliação refeita — teste efetivo: 104 imagens, 117 instâncias. Totais da análise
      de erros (se perguntarem): detecção 95 acertos / 22 FP / 22 FN; segmentação
      101 acertos / 28 FP / 16 FN (conf ≥ 0,25, IoU ≥ 0,5).
- [ ] Notebook Colab com **saídas visíveis** (Item 3 exige): executar e salvar com as
      saídas.
- [ ] Criar o repositório no GitHub e colocar o link no slide final.
- [ ] Preencher nomes do grupo e revisão manual das labels (README, seção 6).
- [ ] Citar na tela ou na descrição: iNaturalist (CC), vídeos do Wikimedia Commons
      (Martin Thurnherr, CC BY-SA 4.0; Tokyo Zoo, CC BY 3.0), YOLOE-26 e Ultralytics.
- [ ] Declarar o uso de IA generativa no relatório (Claude Code).
- [ ] Subir no YouTube como **não listado** e testar o link numa aba anônima.
