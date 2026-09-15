# Prompt para agente de código (Claude Code ou similar)

Cole o texto abaixo como instrução inicial do agente, dentro da raiz do
repositório `wildlife-roadkill-cv` (depois de já ter feito `git clone`).
Ajuste os trechos entre `[colchetes]` antes de colar.

---

```
Contexto do projeto:
Este repositório é um trabalho de pós-graduação (disciplina Visão
Computacional e Reconhecimento de Padrões) que constrói um sistema de
detecção + segmentação de instâncias para 5 espécies de fauna silvestre
brasileira frequentemente atropeladas em rodovias: capivara,
cachorro-do-mato, tamanduá-bandeira, tatu e lobo-guará.

O scaffold já existe: README.md, config/classes.yaml, config/classes_seg.yaml,
scripts/01 a 05 (preparação de dataset, treino de detecção, treino de
segmentação, avaliação, inferência em vídeo), notebooks/main_colab.ipynb,
e docs/ com proposta, template de relatório e roteiro de vídeo.

O que preciso que você faça agora, nesta ordem, PARANDO para eu revisar
entre cada etapa (não avance para a próxima sem eu confirmar):

1. Revise scripts/01_prepare_dataset.py, 02_train_detect.py, 03_train_seg.py,
   04_evaluate.py e 05_infer_video.py quanto a bugs, chamadas desatualizadas
   da API do Ultralytics, e edge cases (ex.: pasta vazia, imagem sem label
   correspondente). Não mude a estrutura de argumentos de linha de comando
   sem avisar, porque o notebook já chama esses scripts com esses nomes.

2. [Depois que eu tiver baixado o dataset real em data/raw/] Rode
   scripts/01_prepare_dataset.py e me mostre o resumo da EDA (contagem por
   classe, resolução média) antes de eu decidir se preciso rebalancear ou
   coletar mais imagens de alguma classe.

3. [Depois que o dataset estiver processado] Configure e rode um treino
   curto de detecção (ex.: 5 épocas, yolov8n) só para validar que o pipeline
   inteiro funciona de ponta a ponta, antes de rodar o treino completo
   (100 épocas) que vai consumir GPU de verdade.

4. Escreva testes simples (pytest) para as funções puras dos scripts que não
   dependem de GPU — especialmente split_dataset() e compute_iou() em
   01_prepare_dataset.py e 04_evaluate.py.

5. NÃO gere dados sintéticos, métricas fictícias ou resultados de treino
   fabricados em nenhuma hipótese — se algo não puder ser executado
   (ex.: falta GPU, falta dataset), diga isso explicitamente e pare, em vez
   de preencher números de exemplo.

6. Ao final de cada etapa, atualize o README.md se algum comando ou caminho
   mudou, para o repositório continuar reproduzível por qualquer integrante
   do grupo.

Restrições:
- Não modifique docs/relatorio_tecnico_template.md nem docs/roteiro_video_pitch.md
  — esses são preenchidos manualmente pelo grupo com resultados reais.
- Todo código deve ter comentários em português, consistente com o resto
  do repositório.
- Se precisar de uma decisão de design (ex.: qual versão do YOLO usar),
  pergunte em vez de assumir.
```

---

## Por que esse prompt é estruturado assim

- **Passos numerados com parada explícita**: evita que o agente rode um
  treino completo (caro, demorado) antes de validar o pipeline com poucas
  épocas.
- **Proibição explícita de fabricar resultados**: crítico em contexto
  acadêmico — métricas inventadas seriam integridade acadêmica violada, não
  só um bug.
- **Restrição sobre o que não mexer**: os documentos que o grupo preenche à
  mão (relatório, roteiro) não devem ser sobrescritos pelo agente.
