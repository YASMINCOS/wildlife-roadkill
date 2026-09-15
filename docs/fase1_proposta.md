# Fase 1 — Proposta do Projeto

**Disciplina:** Visão Computacional e Reconhecimento de Padrões
**Cenário:** Cidades Inteligentes (variante aprovada: prevenção de atropelamento de fauna silvestre)
**Grupo:** _[nomes completos dos integrantes]_

## Problema

Rodovias brasileiras concentram um dos maiores índices de atropelamento de fauna
silvestre do mundo: estimativas do Centro Brasileiro de Estudos em Ecologia de
Estradas (CBEE/UFLA) apontam centenas de milhões de animais mortos por ano nas
estradas do país, com 90% das vítimas sendo pequenos vertebrados e mamíferos
concentrando parte relevante das mortes com maior repercussão pública e
ambiental (inclui espécies ameaçadas de extinção). Um sistema de visão
computacional capaz de detectar e segmentar animais silvestres em imagens de
câmeras de rodovia é o primeiro passo técnico para um futuro sistema de alerta
a motoristas em tempo real — proposta similar a um estudo do ICMC-USP
publicado na *Scientific Reports*, que testou detecção (sem segmentação) de
mamíferos brasileiros com YOLO.

## Classes-alvo

Selecionadas por serem consistentemente reportadas como as mais atropeladas em
estudos de fauna rodoviária no Brasil central e por terem morfologia
distinguível entre si (reduz ambiguidade para o modelo):

1. **Capivara** (*Hydrochoerus hydrochaeris*)
2. **Cachorro-do-mato** (*Cerdocyon thous*)
3. **Tamanduá-bandeira** (*Myrmecophaga tridactyla*) — espécie ameaçada
4. **Tatu** (*Euphractus sexcinctus*, tatu-peba)
5. **Lobo-guará** (*Chrysocyon brachyurus*) — espécie ameaçada

## Fonte dos dados

- **Detecção (bounding box):** datasets públicos do Roboflow Universe com
  imagens de fauna silvestre (ex.: projetos "roadkill" e "wildlife detection",
  licença CC BY 4.0, fonte citada no dataset final). Como esses datasets têm
  espécies predominantemente não-brasileiras, complementaremos com imagens
  livres de direitos autorais das 5 espécies-alvo (bancos como Wikimedia
  Commons/GBIF), citando a origem de cada imagem.
- **Segmentação (máscara/polygon):** a maioria dos datasets públicos de fauna
  só tem bounding box. Vamos anotar manualmente as máscaras de um subconjunto
  representativo (mínimo necessário para atender ao requisito de segmentação)
  usando CVAT ou o próprio anotador do Roboflow, no mesmo domínio das 5
  classes.
- Meta mínima: 300 imagens anotadas no total, com split fixo 70/15/15
  (treino/validação/teste) e seed fixa para reprodutibilidade.

## Ferramenta de anotação

**CVAT** para as máscaras de segmentação (melhor suporte a polygon manual) e
**Roboflow** para organizar/exportar os splits finais em formato YOLO
(detecção) e YOLO-seg (segmentação).

## Próximos passos (Fase 2)

Baixar e consolidar as imagens, rodar a EDA (contagem por classe, resolução,
condições de luz), documentar desbalanceamento e iniciar o fine-tuning do
detector.
