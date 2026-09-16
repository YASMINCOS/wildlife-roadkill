# Passo a passo — testar o projeto e gravar a demonstração

Guia para qualquer integrante do grupo rodar os testes e mostrar o sistema funcionando no vídeo-pitch.
Tudo roda no Mac onde o projeto foi treinado (pasta `~/Documents/visao`), sem precisar treinar de novo.

---

## Parte 1 — Abrir o terminal na pasta do projeto

**Opção A — dentro do IntelliJ:** abra o projeto e vá em **View → Tool Windows → Terminal**
(ou `⌥ F12`). O terminal já abre na pasta do projeto.

**Opção B — app Terminal do Mac:** abra o Terminal e digite:

```bash
cd ~/Documents/visao
```

> Para aumentar a fonte (importante na gravação): `⌘ +` no Terminal do Mac.

---

## Parte 2 — Testar se está tudo funcionando (≈ 1 minuto)

Cole este comando e aperte Enter:

```bash
PAUSA=0 ABRIR=0 bash scripts/demo_video.sh
```

Ele confere os arquivos, roda os testes automáticos e faz a detecção e a segmentação numa foto, **sem
abrir janelas**. O resultado esperado é parecido com isto:

```
=================== 0. Conferindo arquivos necessários ===================
ok     .venv/bin/python
ok     outputs/detect/weights/best.pt
...
device: mps
=================== 1. Testes automáticos (pytest) ===================
...........................................................     [100%]
59 passed
=================== 2. Detecção — caixa + espécie + confiança ===================
image 1/1 .../inat_343605085.jpg: 640x640 1 tamandua_bandeira, 5.7ms
=================== 3. Segmentação — mesma foto, com máscara ===================
image 1/1 .../inat_343605085.jpg: 640x640 1 tamandua_bandeira, 6.7ms
=================== 5. Métricas no conjunto de teste ===================
Modelo                        mAP@0.5  mAP@.5:.95  Precisão  Recall
Detecção (caixas)               0.847       0.744     0.849   0.807
Segmentação (máscaras)          0.875       0.712     0.910   0.846
...
Demonstração concluída.
```

✅ **Se aparecer `59 passed`, `1 tamandua_bandeira` duas vezes e `Demonstração concluída.`, está tudo certo.**

### Se der erro

| Mensagem | O que significa | O que fazer |
|---|---|---|
| `FALTA  outputs/.../best.pt` (ou outro arquivo) | Arquivo gerado no treino/avaliação não está na pasta | Confirme que está em `~/Documents/visao`; os pesos só existem neste Mac (não vão para o GitHub) |
| `No such file or directory: scripts/demo_video.sh` | Terminal está em outra pasta | Rode `cd ~/Documents/visao` e tente de novo |
| `failed` no pytest | Algum teste quebrou | Copie a saída e peça ajuda antes de gravar |
| `device: cpu` | O Mac não usou a GPU (MPS) | Funciona igual, só um pouco mais lento |

---

## Parte 3 — Rodar a demonstração para gravar

1. Deixe preparados: a câmera (se for aparecer) e o terminal com fonte grande.
2. Comece a gravar a tela: **`⌘ Shift 5` → "Gravar tela inteira"** (ou OBS/Loom).
3. No terminal, rode:

```bash
bash scripts/demo_video.sh
```

4. O script **para a cada etapa** e espera você apertar **Enter**. Fale a parte do roteiro
   (`docs/roteiro_falas_pitch.md`) enquanto a imagem está na tela:

| Etapa do script | O que abre | Bloco do roteiro |
|---|---|---|
| 0–1. Arquivos e testes | Saída no terminal (`59 passed`) | Opcional — pode cortar na edição |
| 2. Detecção | Foto do tamanduá com **caixa** e confiança | Bloco 4 — "o detector desenha a caixa…" |
| 3. Segmentação | Mesma foto com **máscara** | Bloco 4 — "agora o modelo de segmentação…" |
| 4. Caixa × máscara | Figura lado a lado | Bloco 4 — "lado a lado fica claro…" |
| 5. Métricas | Tabela no terminal + matriz de confusão | Bloco 5 — resultados |
| 6. Erros | Cachorro-do-mato à noite + tamanduá na armadilha fotográfica | Bloco 5 — falso positivo / falso negativo |
| 7. Vídeo | Vídeo das capivaras com máscaras e IDs | Bloco 6 — vídeo em ação |

5. Feche cada janela de imagem (`⌘ W`) antes de apertar Enter, para a tela não ficar poluída.
6. Para o trecho do **lobo-guará** (Bloco 6), abra o segundo vídeo:

```bash
open outputs/video_lobo_guara/video_lobo_guara.mp4
```

---

## Parte 4 — Mostrar o notebook (Item 3 da entrega)

Abra `notebooks/main_colab.ipynb` no IntelliJ e **role a tela** mostrando as saídas já salvas (logs,
curvas de treino, matrizes de confusão, exemplos de erro e quadros dos vídeos). Não precisa executar de
novo na gravação.

Se quiser **executar o notebook inteiro** (≈ 5 minutos; refaz avaliação e vídeos):

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/main_colab.ipynb
```

---

## Parte 5 — Opcional: vídeo sendo processado ao vivo

Abre uma janela em que as máscaras aparecem em tempo real sobre o vídeo:

```bash
.venv/bin/yolo track model=outputs/segment/weights/best.pt source=data/video_teste.mp4 show=True save=False device=mps
```

Para parar: **`Ctrl C`** no terminal. ⚠️ Teste antes de gravar — se a janela não abrir, use o vídeo pronto
da etapa 7, que já atende o requisito.

---

## Parte 6 — Depois de gravar

1. Confira a duração (**5 a 8 min**) e se **todos os integrantes aparecem/falam**.
2. Suba no YouTube como **não listado**.
3. Abra o link numa **aba anônima** para ter certeza de que funciona.
4. Coloque o link em `docs/relatorio_tecnico.md` (campo "Vídeo-pitch").

---

## ⚠️ Sobre mensagens do Git no IntelliJ

Enquanto o repositório próprio do projeto não for criado, **não use `git add`/commit pelo IntelliJ nesta
pasta**: o Git que ele enxerga é o da pasta pessoal (outro projeto), não o deste trabalho. Mensagens como
`Run git add ... to see line counts` podem ser ignoradas.
