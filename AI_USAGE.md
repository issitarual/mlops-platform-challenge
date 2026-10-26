# Uso de IA

Registro de como a IA entrou no projeto: o que ela fez, como o trabalho foi
verificado e onde ela errou. Cada decisão de design foi minha; a IA propôs
alternativas, implementou e revisou. As alternativas eram verificadas e revisadas por mim.

## Ferramentas

- **Claude (Anthropic)**, no app do Claude, com acesso a um ambiente Linux isolado na
  nuvem (shell, Python, Docker sem acesso ao Docker Hub) e, por uma ponte do app
  desktop, à pasta do projeto no meu computador. Sem acesso ao meu git: **todo commit
  foi feito por mim**, depois de revisar.
- Ferramentas de verificação usadas pela IA e por mim: `ruff`, `mypy` (Linux, Windows e
  macOS), `pytest`, um servidor `mlflow server` 3.17.2 real e o CI no GitHub Actions.

## Como foi usada

- **Planejamento:** épicos E0–E10 com entregáveis e critérios de pronto; marcos M1 a
  M3. A IA propôs; eu reordenei e cortei escopo.
- **Decisões:** para cada escolha relevante, a IA apresentava 2 a 3 alternativas com
  prós, contras e o que o mercado usa, com uma recomendação. **A decisão era minha** e
  ficou registrada no `DECISIONS.md` (ADR-001 a ADR-021).
- **Implementação:** a IA escrevia numa cópia de trabalho, rodava lint, tipos e testes
  e só então copiava para a minha pasta. Eu revisava, rodava localmente e commitava.
- **Revisão crítica:** ao fim de cada épico, pedi uma revisão "como Staff de MLOps",
  em formato 🔴/🟡/🟢 com veredito. Os bloqueantes eram corrigidos antes do push.
- **Verificação:** o que a IA não conseguia rodar (Docker Hub, `ghcr.io`, Windows) eu
  rodei no meu computador ou ficou para o CI. A regra combinada: nada é dado como
  verificado sem ter rodado.

## Prompts que funcionaram

- *"Me manda opções com prós e contras, o que é usado no mercado, quais são as melhores
  práticas."* Transforma "faça X" em uma decisão que eu consigo defender.
- *"Essas são decisões que um Staff de MLOps faria?"* Obriga a justificar a recomendação
  com o critério da vaga, e às vezes mudou a resposta.
- *"Faça uma revisão crítica de tudo que foi alterado, como um Staff de MLOps, seguindo
  boas práticas, segurança e qualidade de código."* Encontrou falhas reais no próprio
  código gerado pela IA (exemplos abaixo).
- *"Não commite nada. Eu reviso e commito."* Mantém o histórico do git sob meu controle.
- Colar a saída completa de um comando que falhou, em vez de descrever o erro.

## Sugestões rejeitadas ou corrigidas

### Corrigidas por mim

- **Fronteira de confiança do MLflow e skew de ambiente (planejamento).** O desenho
  inicial não separava o que roda dentro do container de treino do que fala com o
  MLflow, e treino e serving podiam acabar em ambientes diferentes. Ficou: o container
  não tem rede nem credenciais, o runner salva o modelo offline, o host verifica e
  registra (ADR-006); treino e serving usam a mesma imagem (ADR-005).
- **Restrição a scikit-learn.** A primeira proposta de serialização segura servia só
  para scikit-learn. Questionei a liberdade do cientista quanto ao tipo de modelo, e o
  resultado foram os níveis de confiança do ADR-017 (skops por padrão, pickle só com
  opt-in explícito, formatos nativos no desenho).
- **Champion e challenger.** Notei que a imagem por versão de modelo não tratava
  champion e challenger; virou o ADR-013 (gate de promoção) e a seção no ADR-007.
- **Versionamento de dados.** Perguntei por que não versionar a fonte de dados a cada
  alteração; a discussão virou o ADR-014 (hash por arquivo no código, armazenamento
  versionado no desenho).
- **Tom do DECISIONS.md.** Recusei uma seção que se transformou em "Premissas adotadas".

- **Cache de build no CI.** Questionei por que não usar cache de camadas no CI. A IA
  manteve a recomendação, mas condicionou a decisão a um dado: com o job de
  integração em 3m44s, e com o cache exigindo trocar o caminho de build que está
  sendo testado, ficou sem cache. O critério para revisitar ficou registrado (ADR-022).

### Erros da IA, encontrados na verificação ou na revisão

- **Recomendação que contradizia uma decisão já tomada.** Para o E3, a IA recomendou
  MLflow em SQLite local sem servidor, sem reler o ADR-008 (servidor via compose). Ela
  mesma apontou o conflito antes de implementar, e a recomendação foi revista.
- **Três bloqueantes no próprio código do E3**, achados na revisão crítica: nomes de
  métrica que o MLflow recusa só no fim do treino (e de forma diferente no Windows),
  parâmetros sem validação até o registro final, e o run podendo ficar `RUNNING` para
  sempre após um Ctrl+C. Todos corrigidos e cobertos por testes.
- **Regra de nomes suposta, depois verificada.** Em vez de confiar na memória, a regra
  foi testada contra o validador do próprio MLflow: `"rmse (test)"` e `"f1@0.5"` são
  recusados, e `:` só é aceito fora do Windows.
- **Código que quebrava no Windows.** `os.getgid` não existe no Windows; o mypy no meu
  computador pegou. Desde então o mypy roda para as três plataformas.
- **Hash dependente do sistema operacional.** O script de download gravava o CSV com
  fim de linha do sistema, mudando o SHA-256 entre Windows e Linux. Corrigido fixando
  `\n` e verificando o hash em memória antes de gravar.
- **`Distribution.files` esconde arquivos ausentes.** Na primeira versão do self-check
  da imagem, um arquivo apagado passava despercebido; um teste pegou, e a verificação
  passou a ler o `RECORD` diretamente.
- **Código de erro do MLflow suposto.** A IA assumiu que alias inexistente voltava como
  "não encontrado"; o teste contra o MLflow real mostrou `INVALID_PARAMETER_VALUE`, e o
  tratamento foi ajustado.

- **Bypass da política de carga no próprio código do E4.** A primeira versão aceitava
  qualquer modelo com flavor `sklearn` em skops. Um `MLmodel` adulterado podia manter
  esse flavor e trocar o `loader_module` do `python_function` ou embutir código. A
  regra passou a exigir também o loader `mlflow.sklearn` e nenhum código embutido, com
  um teste para cada adulteração.
- **Três bugs na API que passaram pela revisão por leitura e pelos testes.** Só
  apareceram quando a IA rodou a API com entradas adversas: corpo em chunks acima do
  limite voltava 400 em vez de 413; `NaN` no JSON chegava ao modelo e virava 500; e
  um 500 não entrava nas métricas nem levava o `X-Request-ID`, o que deixaria um
  alerta de taxa de erro cego. Os testes cobriam só o limite com `Content-Length` e
  valores finitos. Corrigidos, com testes de regressão.
- **Falha de segurança no desenho da própria IA (E4).** O `serve` lia a chave do
  ambiente e a permissão de pickle do `manifest.json` e do `run.json` baixados do
  artifact store, sem verificar. Quem tivesse escrita só no bucket trocaria o lock por
  um com pacotes maliciosos. A âncora de confiança passou a ser o banco do MLflow
  (tags do run), e o lock só é usado se o hash bater (ADR-020).

- **Saída da validação engolindo texto do usuário (desde o E2).** Os problemas eram
  impressos com markup do Rich sem escapar: `'pandas[performance]=2'` aparecia como
  `'pandas=2'`, justamente na linha que o cientista precisa corrigir. Achado no E5 ao
  escrever o teste da saída do `validate`; corrigido escapando e redigindo a mensagem.
- **Limites mínimos do runtime nunca testados.** Ao relaxar `pandas>=3.1.0` para 2.2,
  a IA rodou os testes com as versões mínimas de fato: `pandas` 2.2.0 emite um
  `DeprecationWarning` no import que quebra a suíte (avisos são erro). O mínimo ficou
  em 2.2.2, e um job do CI passou a testar as versões mínimas.

- **Validação que quebrava com arquivos reais do Windows e varria código que não roda
  (E5).** A revisão crítica, rodando entradas reais, achou: traceback com pasta de
  projeto inexistente; traceback com `requirements.txt` em UTF-16 (o `>` do Windows
  PowerShell 5) e com `train.py` em latin-1; erro falso com BOM, que o Python e o `uv`
  aceitam. E o scan de todos os `.py` do projeto transformava o `pytest` de `tests/`
  em problema, o que inviabilizava o `--strict` no CI. Os 299 testes estavam verdes.
  Ficou: leitura a partir dos bytes, como o Python lê; erro claro de codificação;
  análise só do que o `train.py` importa.

- **Dois bugs do produto achados pelo teste ponta a ponta (E6).** Escrever o teste
  mostrou que o `serve` só removia o container com Ctrl+C (SIGTERM o deixava rodando).
  Rodá-lo no meu Windows mostrou que a CLI quebrava com a saída redirecionada: o
  Windows usa cp1252 em pipes, e o MLflow imprime emojis no fim do run. Nenhum teste
  unitário pegaria isso; a correção veio com testes de regressão. Na mesma revisão
  saiu um teste unitário antigo que, pelo nome, testava "comando não implementado",
  mas na prática tentava falar com o MLflow.

- **Decisão registrada como feita sem ter sido implementada (E2 a E7).** O ADR-016 e o
  plano do sprint diziam que o runner já gravava estatísticas das features de treino no
  manifesto (E2). Não gravava: a IA registrou a intenção como fato. Apareceu no E7,
  ao procurar essa baseline para o drift, e a IA avisou antes de seguir. A baseline foi
  implementada (agora dentro da pasta do modelo, coberta pelo hash), e o ADR reescrito.
  Desde então, cada afirmação de "implementado" nos docs é conferida no código.
- **Coleta do Prometheus que não funcionaria no Linux (E7).** A primeira versão, a
  recomendação aceita, raspava `host.docker.internal`. Na revisão, a IA viu que a API é
  publicada só em `127.0.0.1` do host, que o gateway do Docker no Linux não alcança. No
  Docker Desktop isso deve passar, então o teste na minha máquina provavelmente não pegaria. Passou para
  uma rede só entre a API e o Prometheus, com descoberta por DNS (ADR-016).

- **Serving sem limite de threads (E4, achado no E7) e a primeira correção, que
  também estava errada.** O treino limitava as threads das bibliotecas nativas pela cota
  de CPU, mas o serving não, e a IA não percebeu a diferença ao escrever o E4. Com 28
  CPUs no host e 2 na cota, o teste de carga deu 17 req/s e p95 de 2,3 s, contra 101
  req/s e p95 de 56 ms em outro cenário. Rodei o teste e a IA estranhou os números,
  comparando os cenários, e levantou a hipótese (CFS throttling). Ela recomendou
  `cpus ÷ workers` threads e eu aceitei. A medição desmentiu: com 2 workers (1 thread)
  foi de 17 para 335 req/s, mas com 1 worker (2 threads) o `batch` ficou em 9,6 req/s.
  A IA reviu a recomendação, expôs os números lado a lado e propôs 1 thread por
  predição, que adotei: com 1 worker o `batch` foi para 163 req/s e p95 de 170 ms. Lição: a recomendação veio de raciocínio, e só a medição em mais
  de uma configuração achou o furo (ADR-020).
- **Dados de carga enviesados na própria IA (E7).** O script de Locust usava as
  primeiras 5000 linhas do CSV, que na Califórnia estão ordenadas por região. O
  dashboard mostrou o alerta de drift disparado sem nenhum deslocamento proposital.
  A IA conferiu nos dados antes de concluir: PSI de 1,16 e 1,02 em Latitude e
  Longitude com as primeiras linhas e ≈ 0 com amostra aleatória. O drift estava certo,
  o teste é que estava errado. Também saíram do mesmo print: painel de erro sem dado
  quando não há 5xx e "Model served" mostrando o valor `1` em vez do modelo.

- **Funcionalidades descritas como prontas sem estar (E8 e E9).** Ao desenhar a arquitetura
  "hoje" e ao escrever o README, a IA afirmou duas coisas que o código não tem: uma rota
  `/v1/predict/batch` (só existe `/v1/predict`, com várias linhas por requisição) e um gate de
  promoção que compara champion e candidato com `--force` (o `promote` só move o alias e
  registra quem, quando e o que substituiu; o `TrainOutput` nem tem `primary_metric`). Ambas
  vieram do plano e do ADR, não do código. A IA achou os dois ao conferir cada comando e cada
  rota na CLI e no runtime antes de documentar, e corrigiu diagramas, README e DECISIONS.
  O ADR-013 agora diz que o gate não foi implementado e o item foi para "Com mais tempo".
  Lição: documentação de estado atual se confere no código, não no plano.

- **Mermaid, diagramas emaranhados e badges de passo (E8).** O primeiro desenho da AWS foi
  em Mermaid. Com `direction` aninhado em subgrafos, as setas colapsaram para a borda do
  subgrafo, e o diagrama 3 virou um emaranhado de nós. A IA simplificou duas vezes antes de
  concluir que o formato era o problema, e a troca para diagramas por passo no estilo
  draw.io (um passo por imagem) foi proposta por mim, a partir do artigo do Jarvis da OLX.
  Depois, o selo "STEP N" ficou longe do conteúdo, os diagramas não se identificavam e um
  cartão de CI estava no passo errado; vi cada imagem e pedi o ajuste. 
- **Batch Transform some na conversão (E8).** Ao converter o diagrama da AWS em passos, a IA
  deixou de fora o Batch Transform, que era a única inferência em lote do desenho. Notei na
  revisão e o cartão voltou para o passo 4. O lote continua só no desenho.

- **Pickle autoconcedido pelo `train.py` (revisão externa, E9).** Uma revisão externa, trazida
  pela Laura para avaliar, apontou que o serving liberava pickle com
  base na tag `mlship.serialization`, vinda do `result.json`, que o próprio `train.py` pode
  escrever (roda no mesmo processo do runner). A IA que escreveu o código não tinha visto: a
  decisão de confiança estava no código do cientista, contra o ADR-017. Conferi a cadeia no
  código antes de aceitar (o achado procede), e a correção ficou menor que a proposta: o host
  grava a tag `mlship.allow_pickle` a partir da flag, o treino sem flag recusa serialização
  diferente de skops, e o serving só libera pickle com a tag, a plataforma ligada e a
  serialização registrada. A opção `serve --allow-pickle` explícita ficou em "Com mais
  tempo", por decisão da Laura. Quatro testes novos cobrem o caso (resultado forjado
  no treino, resultado permitido com a flag, tag `false` no serving e plataforma desligada);
  a revisão seguinte notou que nenhum cobria a tag ausente, e acrescentei esse. Rodei unitários, mypy e ruff, mas não o e2e com Docker.
- **Achados da mesma revisão que rebaixei ou ajustei.** O vetor de `.pth` (dependência que
  executa código antes da política de carga) é real, e minha justificativa inicial era
  frágil: dizer que uma dependência maliciosa "já executa código no `import`" não vale para o
  caminho skops do serving, porque o `.pth` roda sem import (apontado na revisão seguinte).
  Recusar `.pth` quebraria pacotes legítimos e é decisão de allowlist, então virou registro
  no ADR-017, não código. O ADR-017 também afirmava que o serving não tem rede e
  que o `serve` avisa de modelos pickle, e nenhuma das duas coisas é verdade: corrigi o
  texto. O split do exemplo e do template dependia da seed do modelo; agora é fixo
  (`SPLIT_SEED`), com um baseline de média.

### Achados em que a IA ajudou a diagnosticar

- **Imagem corrompida com o disco cheio.** O treino morria com `SIGBUS` e sem log. A IA
  comparou o `typing_extensions.py` dentro da imagem com o arquivo correto e concluiu
  que o build terminou "com sucesso" gravando arquivos truncados. Daí veio o self-check
  da imagem (ADR-005) e as mensagens legíveis por sinal (ADR-018).
- **Telemetria do MLflow.** Durante os testes, o proxy do ambiente da IA registrou
  conexões recusadas para `config.mlflow-telemetry.io`: o MLflow 3 envia telemetria de
  uso por padrão. Ficou desligada na CLI, na imagem e no servidor (ADR-019).

## Limites do uso de IA

- As horas do "Tempo gasto" (DECISIONS.md) são estimativas por faixa, sugeridas pela IA a
  partir das datas dos commits e da duração das sessões, e ajustadas por mim; a IA não
  tem como medir o trabalho feito fora da conversa.
- O ambiente da IA não alcançava o Docker Hub nem o `ghcr.io`: o build real das
  imagens, o digest do servidor MLflow e os testes no Windows foram feitos por mim e
  pelo CI.
