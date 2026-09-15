# Fontes de dados

## Dataset efetivamente usado

**Imagens — iNaturalist** (https://www.inaturalist.org), baixadas por
`scripts/00_download_inaturalist.py`:

- observações *research grade* (espécie confirmada pela comunidade) com a anotação
  "Evidence of Presence = Organism" (exclui fotos só de fezes, pegadas ou ossos);
- apenas fotos com licença CC0, CC BY, CC BY-SA ou CC BY-NC; uma foto por observação;
- autor, licença e link de **cada imagem** em `data/ATRIBUICAO_imagens.csv` (citar no relatório).
- download de 13/09/2026: 900 imagens (180 por classe) — licenças CC BY-NC 762, CC BY 103,
  CC BY-SA 19, CC0 16. Por causa das imagens **CC BY-NC**, o dataset e os pesos treinados
  só podem ser usados para fins **não comerciais** (ok para o trabalho acadêmico); as
  imagens **CC BY-SA** exigem que derivados delas sejam compartilhados com a mesma licença.

| Classe | Táxon iNaturalist |
|---|---|
| capivara | 74442 — *Hydrochoerus hydrochaeris* |
| cachorro_do_mato | 42087 — *Cerdocyon thous* |
| tamandua_bandeira | 47107 — *Myrmecophaga tridactyla* |
| tatu | 47083 — *Euphractus sexcinctus* |
| lobo_guara | 42091 — *Chrysocyon brachyurus* |

**Labels — pseudo-labels YOLOE-26** (`scripts/00b_autolabel_yoloe.py`): o modelo
open-vocabulary localiza o animal (caixa + máscara) a partir de prompts de texto; a
classe vem do táxon do iNaturalist. Imagens sem detecção confiável são descartadas
(lista em `outputs/autolabel/report.json`). Limitação a declarar: labels não foram
anotadas manualmente — erros do YOLOE (contorno impreciso, animal parcialmente
detectado, outro animal da foto rotulado como a espécie) viram ruído no treino.

**Imagens ignoradas (GIF com extensão `.jpg`):** 15 das 713 imagens processadas vieram do
iNaturalist em formato GIF, apesar da extensão `.jpg` (10 no treino, 1 na validação, 4 no
teste). O Ultralytics as descarta como formato não suportado, então não entraram no treino
nem nas métricas; `scripts/04_evaluate.py` aplica o mesmo filtro na análise de erros.
Conjunto de teste efetivo: **104 imagens, 117 instâncias**.

**Vídeos de teste — Wikimedia Commons** (convertidos para MP4 sem áudio):

| Arquivo local | Original | Autor | Licença | Duração |
|---|---|---|---|---|
| `data/video_teste.mp4` | [Capybaras - Wasserschweine.webm](https://commons.wikimedia.org/wiki/File:Capybaras_-_Wasserschweine.webm) — capivaras na Fazenda San Francisco, Miranda-MS | Martin Thurnherr | CC BY-SA 4.0 | 89 s |
| `data/video_lobo_guara.mp4` | [(Inhabits South America) Maned Wolf (Ueno Zoological Gardens in Tokyo, Japan).webm](https://commons.wikimedia.org/wiki/File:(Inhabits_South_America)_Maned_Wolf_(Ueno_Zoological_Gardens_in_Tokyo,_Japan).webm) | Tokyo Zoo | CC BY 3.0 | 84 s |

Os vídeos não são versionados (`*.mp4` no `.gitignore`); para baixar de novo, use os links acima.

---

## Levantamento inicial (Fase 1)

## Datasets públicos identificados (Roboflow Universe)

| Dataset | Imagens | Classes | Licença | Link |
|---|---|---|---|---|
| roadkill | 778 | animal (genérica) | CC BY 4.0 | https://universe.roboflow.com/citra-zhzbq/roadkill-t9w1h-yplts |
| Wildlife detection | — | fauna silvestre diversa | verificar na página | https://universe.roboflow.com/wildlife-2qlrs/wildlife-detection-nvoaq |
| Project one Wildlife system | 2.806 | cão, gato, ave, vaca, veado, raposa, javali, etc. (14 classes) | verificar na página | https://universe.roboflow.com/project-one-wildlife-monitoring-system/project-one-wildlife-system-lsden |
| Busca geral "wild animals" | — | dezenas de projetos (veado, guaxinim, coiote, lince, etc.) | variável | https://universe.roboflow.com/search?q=class%3Awild+animals |

**Limitação importante:** nenhum desses datasets é focado especificamente em
fauna brasileira (capivara, tamanduá-bandeira, cachorro-do-mato, tatu,
lobo-guará) e a maioria só tem anotação de bounding box, não máscara de
segmentação. Isso é esperado — documentar essa limitação no relatório
técnico (seção "Limitações").

## Complemento necessário

1. **Imagens das 5 espécies-alvo:** buscar em bancos de imagem livres de
   direitos autorais (Wikimedia Commons, GBIF, iNaturalist — verificar
   licença de cada imagem individualmente antes de usar) até atingir
   representatividade mínima por classe.
2. **Máscaras de segmentação:** anotar manualmente (CVAT) um subconjunto
   representativo de cada classe. Não é necessário anotar todas as 300+
   imagens em polygon — um subconjunto balanceado (ex.: 40-60 por classe)
   já atende ao requisito de "segmentação no mesmo domínio" e permite a
   comparação caixa × máscara pedida na Fase 3.
3. **Vídeo de teste:** gravar ou conseguir um trecho de vídeo real (câmera
   de rodovia, parque, ou filmagem própria) com pelo menos uma das espécies,
   ≥30 segundos, para o requisito de inferência em vídeo.

## Referência acadêmica citável

CBEE/UFLA (Centro Brasileiro de Estudos em Ecologia de Estradas) — dados de
atropelamento de fauna no Brasil: http://cbee.ufla.br

Pesquisa relacionada (ICMC-USP, detecção sem segmentação): reportagem da
Agência FAPESP,
https://agencia.fapesp.br/sistema-utiliza-inteligencia-artificial-para-detectar-animais-selvagens-na-pista-e-evitar-acidentes/51095
