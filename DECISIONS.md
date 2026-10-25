# Decisões

Formato ADR-lite: contexto → decisão → alternativas descartadas → consequências.
Status: **Aceita** (em vigor) ou **Proposta** (aguardando confirmação).

## Índice

| # | Decisão | Status |
| --- | --- | --- |
| [001](#adr-001--cli-em-python-orquestrando-docker) | CLI em Python orquestrando Docker | Aceita |
| [002](#adr-002--contrato-mínimo-do-cientista) | Contrato mínimo do cientista | Aceita |
| [003](#adr-003--workspace-uv-com-três-pacotes) | Workspace `uv` com três pacotes | Aceita |
| [004](#adr-004--camadas-com-ports-e-adapters-aplicadas-por-pacote) | Camadas com ports e adapters, aplicadas por pacote | Aceita |
| [005](#adr-005--um-ambiente-para-treino-e-serving-identificado-por-todas-as-entradas) | Um ambiente para treino e serving, identificado por todas as entradas | Aceita |
| [006](#adr-006--mlflow-skinny-no-runtime-e-modelo-salvo-offline) | `mlflow-skinny` no runtime e modelo salvo offline | Aceita |
| [007](#adr-007--imagem-de-serving-por-versão-de-modelo) | Imagem de serving por versão de modelo | Aceita |
| [008](#adr-008--mlflow-local-com-sqlite) | MLflow local com SQLite | Aceita |
| [009](#adr-009--docker-sdk-para-python) | Docker SDK para Python | Aceita |
| [010](#adr-010--dados-do-exemplo-verificados-por-hash) | Dados do exemplo verificados por hash | Aceita |
| [011](#adr-011--limites-de-reprodutibilidade) | Limites de reprodutibilidade | Aceita |
| [012](#adr-012--versões-das-ferramentas-fixadas) | Versões das ferramentas fixadas | Aceita |
| [013](#adr-013--champion-e-challenger-com-gate-de-promoção) | Champion e challenger com gate de promoção | Aceita (implementação P2) |
| [014](#adr-014--versionamento-de-dados-por-referência) | Versionamento de dados por referência | Aceita |
| [015](#adr-015--versionamento-de-ponta-a-ponta-e-de-modelos) | Versionamento de ponta a ponta e de modelos | Aceita |
| [016](#adr-016--observabilidade-do-serving-e-referência-para-drift) | Observabilidade do serving e referência para drift | Aceita (dashboard P1, referência P2) |
| [017](#adr-017--formatos-de-modelo-por-nível-de-confiança) | Formatos de modelo por nível de confiança | Aceita (níveis 1 e 2; nível 3 no desenho) |
| [018](#adr-018--treino-isolado-e-contrato-com-o-runner) | Treino isolado e contrato com o runner | Aceita |
| [019](#adr-019--lineage-registro-e-promoção) | Lineage, registro e promoção | Aceita |
| [020](#adr-020--serving-da-versão-promovida) | Serving da versão promovida | Aceita |
| [021](#adr-021--validação-do-projeto-lock-e-init) | Validação do projeto, lock e `init` | Aceita |
| [022](#adr-022--estratégia-de-testes-e-ci) | Estratégia de testes e CI | Aceita |

---

## Premissas adotadas

| Tema | Premissa |
| --- | --- |
| Entrada do cientista | `train.py` + `requirements.txt`; os dados ficam num diretório separado, informado no `train` |
| Execução | Uma máquina com Docker roda todo o fluxo local; produção é descrita no desenho AWS |
| Ambiente das imagens | Linux amd64, Python 3.12 |
| Modelos | Executam em CPU e cabem na memória de uma instância |
| Validação antes do treino | Análise do código sem executá-lo (AST); a resolução de dependências consulta o índice de pacotes (PyPI) |
| Exemplo | California Housing, regressão com scikit-learn |
| Prazo | 7 dias corridos a partir de 21/10, com entrega planejada para 27/10 |

---

## ADRs

### ADR-001 — CLI em Python orquestrando Docker

- **Contexto:** o cientista entrega só código Python e precisa chegar a uma API sem
  escrever infraestrutura.
- **Decisão:** uma CLI (`mlship`) com o fluxo `init` → `validate` → `train` → `runs` →
  `promote` → `serve`, que orquestra containers Docker e um MLflow local via `docker compose`.
- **Alternativas descartadas:** orquestrador (Airflow, Kubeflow) ou Kubernetes local:
  peso operacional desproporcional para o escopo; ficam no desenho de produção.
- **Consequências:** a única exigência na máquina do usuário é `uv` + Docker. Cada
  comando local tem um equivalente gerenciado na AWS (ver `docs/architecture`).

### ADR-002 — Contrato mínimo do cientista

- **Contexto:** cada exigência a mais no contrato é atrito para quem usa a plataforma.
- **Decisão:** `train.py` com `train(data_dir: Path, params: dict) -> TrainOutput(model,
  metrics, input_example)` + `requirements.txt`. `TrainOutput` vem do `mlship-sdk`, sem
  dependências. O modelo precisa ter `.predict()`, com o pré-processamento dentro dele.
  Campos **opcionais** `primary_metric` e `direction` (`"minimize"` ou `"maximize"`)
  habilitam o gate de promoção (ADR-013); entram no E3.
- **Critério para o contrato:** só se pede o que a ferramenta não consegue inferir **e**
  que muda o comportamento dela. Limitações da ferramenta não viram exigências do
  contrato: qualquer objeto com `.predict()` é aceito, e o formato de serialização é
  escolhido pela ferramenta conforme o nível de confiança (ADR-017).
- **Alternativas descartadas:**
  - classe base para herdar: acopla o código do usuário à ferramenta;
  - arquivo de configuração YAML: mais um formato para aprender;
  - Dockerfile do usuário: transfere a responsabilidade de segurança e
    reprodutibilidade para ele;
  - campo `task_type` (classificação, regressão...): não muda o comportamento da
    ferramenta (o gate usa a métrica declarada), o tipo de saída é inferido da
    assinatura do modelo, a taxonomia nunca fecha (multirrótulo, ranking, séries) e
    um valor declarado pode contradizer o modelo real.
- **Consequências:** o `input_example` permite inferir o schema da API e fazer smoke
  test sem pedir nada além do treino. O tipo de saída inferido da assinatura vai para o
  manifesto, útil para monitoramento. Parâmetros chegam como string pela CLI; o
  `train()` converte.

### ADR-003 — Workspace `uv` com três pacotes

- **Contexto:** o código roda em três lugares com dependências diferentes: a CLI no
  host, o runner no container de treino e a API no container de serving. O que vai para
  a imagem divide o ambiente com o `requirements.txt` do cientista.
- **Decisão:** workspace `uv` com `packages/sdk` (zero dependências), `packages/runtime`
  (runner + serving, instalado na imagem) e `packages/cli` (host).
- **Alternativas descartadas:**
  - pacote único com extras: extras controlam o que se instala, não o que o código
    importa;
  - repositórios separados: isolamento total, mas excessivo para uma pessoa em 7 dias
    e contrário ao formato de entrega (um repositório).
- **Consequências:** a imagem instala só o `runtime`, o que reduz a chance de conflito
  com o cientista. Em desenvolvimento há um lock e um venv compartilhados, e a
  [documentação do uv](https://docs.astral.sh/uv/concepts/projects/workspaces/) avisa
  que não há isolamento de imports entre membros: por isso existe o teste de
  arquitetura (ADR-004). Custo: o build da imagem precisa gerar o wheel do `runtime`.

### ADR-004 — Camadas com ports e adapters, aplicadas por pacote

- **Contexto:** a CLI integra Docker, `uv` e MLflow; os casos de uso precisam ser
  testáveis sem eles.
- **Decisão:**
  - **CLI:** `controllers → services → domain`. Os services dependem de interfaces
    (`Protocol` em `services/ports.py`), implementadas por `repositories/`
    (persistência: MLflow) e `adapters/` (integrações: Docker, `uv`, hashing). Um único
    composition root (`bootstrap.py`) monta tudo.
  - **Serving:** convenções do FastAPI (`api/v1`, `api/deps.py`, `core/lifespan.py`,
    `engines/`, `services/`, `schemas/`).
  - **Runner:** módulos simples, sem camadas: é um programa curto, linear e de execução única.
  - **SDK:** um arquivo.
  - Um teste próprio (`tests/unit/test_architecture.py`) verifica a direção das
    dependências via AST e reporta todas as violações de uma vez.
- **Alternativas descartadas:**
  - camadas clássicas sem interfaces: Docker e `uv` virariam "repository" sem ser
    persistência, e testar services exigiria mockar bibliotecas;
  - clean/hexagonal completo com container de injeção de dependências: cerimônia
    excessiva para o escopo;
  - organização por funcionalidade: gera pastas quase vazias num projeto deste tamanho;
  - `import-linter` em vez do teste próprio: mais conhecido, mas é uma dependência a
    mais para uma regra que cabe em um arquivo.
- **Consequências:** services testados com fakes, CI unitário rápido e sem rede. Cada
  adapter tem um par AWS (MLflow local → SageMaker; Docker local → SageMaker
  Training/ECS). Custo: mais arquivos e uma camada de indireção na CLI.

### ADR-005 — Um ambiente para treino e serving, identificado por todas as entradas

- **Contexto:** diferença de ambiente entre treino e serving é uma das causas clássicas
  de skew. E um cache de imagens só é correto se a chave muda sempre que qualquer
  entrada do build muda.
- **Decisão:**
  - **lock no host:** `uv pip compile` resolve o `requirements.txt` do cientista junto
    com as dependências do runtime, com hashes, só wheels (`--no-build`), para Python
    3.12 e Linux x86_64 (`x86_64-manylinux_2_28`). O lock existe antes do build;
  - **lock reutilizado:** o resultado fica em `mlship.lock`, ao lado do
    `requirements.txt`, com o hash das entradas na primeira linha. Enquanto
    requirements, runtime e alvo não mudam, o lock é reaproveitado em vez de
    resolvido de novo; assim, o mesmo projeto não pega versões novas por acaso e o
    lock pode ser commitado e compartilhado (como `uv.lock` ou `poetry.lock`);
  - **lock íntegro:** a segunda linha guarda o hash do corpo. Um lock só é
    reaproveitado se o corpo bate com esse hash; um lock truncado, com conflito de
    merge ou editado à mão é resolvido de novo, e a ferramenta avisa. A gravação é
    atômica (arquivo temporário + rename), então nunca existe um lock pela metade;
  - **alvo coerente:** plataforma do Docker, plataforma do lock, versão do Python e
    imagem base são validadas juntas ao carregar a configuração (ex.: imagem arm64 com
    lock x86_64, ou base sem digest, falham na hora, com mensagem clara);
  - **runtime na imagem:** os wheels do `mlship-runtime` e do `mlship-sdk` são gerados
    localmente e instalados com `--no-deps` (as dependências deles já estão no lock);
  - **imagem base:** `python:3.12-slim` (Debian, glibc) fixada pelo digest do índice
    multiarquitetura (`sha256:a0f57064…6704`, consultado no Docker Hub em 23/10), com
    build em múltiplos estágios. No estágio de build, o `pip` da própria imagem instala
    o lock com `--require-hashes --only-binary=:all: --no-deps` e os wheels da
    ferramenta sem índice; depois o `pip` é removido. A imagem final tem só o
    virtualenv, usuário não-root (UID 10001) e nenhuma ferramenta de build;
  - **chave do ambiente (tag da imagem):** hash de **todas** as entradas do build:
    lock, código-fonte do runtime e do sdk, digest da base, template do Dockerfile,
    versão do Python e plataforma. Usa o código-fonte, não o wheel, porque wheels podem
    carregar datas no arquivo e mudariam o hash a cada build;
  - **imagem verificada antes do uso:** logo após o build, um self-check roda na
    própria imagem, com o mesmo isolamento do treino. Ele confere o sha256 de cada
    arquivo instalado contra o `RECORD` que o `pip` gravou e importa os módulos
    nativos do runner. Se falhar, a imagem é apagada (nunca fica no cache) e o erro
    lista os arquivos afetados. Imagens reaproveitadas do cache não são verificadas de
    novo;
  - a mesma imagem serve ao treino e é a base da imagem de serving (ADR-007).
- **Alternativas descartadas:**
  - self-check como último `RUN` do Dockerfile: mais simples, mas verifica o
    sistema de arquivos do build, não a imagem que vai ser usada;
  - verificar também a cada uso do cache: protege contra adulteração local da imagem,
    mas custa segundos a cada `train`; em produção isso é papel da assinatura e do
    digest no registry (desenho);
  - tag igual só ao hash do lock: mudanças no runtime, na base ou no template
    reaproveitariam a imagem antiga do cache, um erro silencioso de reprodutibilidade;
  - sem cache: correto, mas minutos a cada `train`;
  - cache de camadas do Docker como única garantia: heurístico, e a tag não serviria
    para lineage;
  - resolver dentro do `docker build`: o lock só existiria depois do build, e não
    daria para consultar o cache antes de construir;
  - resolver num container de build: mais lento e exigiria Docker até no `validate`;
  - publicar o runtime num índice de pacotes: é o caminho de produção (AWS
    CodeArtifact, no desenho), inviável no prazo;
  - montar o código do runtime na execução: a imagem deixaria de ser autossuficiente;
  - Alpine: usa musl, muitos wheels não funcionam e pacotes compilariam do
    código-fonte, quebrando o "só wheels";
  - distroless ou Chainguard: superfície menor, mas com restrições de versão do Python
    e de depuração; ficam em "Com mais tempo" para o estágio final.
- **Consequências:**
  - mesmas entradas, mesma imagem; qualquer mudança gera outra imagem;
  - conflitos com o runtime aparecem na resolução, antes do treino. Versões não
    fixadas pelo cientista viram aviso, não erro: o lock fixa e informa;
  - a imagem base tem shell e gerenciador de pacotes; o Trivy no CI mede as
    vulnerabilidades;
  - o `pip` da base dispensa copiar o `uv` para a imagem: um binário a menos para
    fixar e verificar;
  - **verificado (E1):** lock real do exemplo com 56 pacotes, determinístico em duas
    resoluções; a política de instalação do Dockerfile simulada fora do Docker
    (instalação, `pip check` sem conflitos, treino e modelo salvo e carregado sem rede);
    o adapter do Docker SDK construindo, encontrando e rotulando imagens. O build com a
    imagem base real roda no job de integração do CI;
  - **limites conhecidos, registrados:**
    - opções de supply chain no `requirements.txt` (`--index-url`, `-e`, `git+`, URLs,
      caminhos locais, `--hash` inline) são barradas pela validação antes de o `uv`
      ver o arquivo (ADR-021);
    - o índice de pacotes é implícito (PyPI). Variáveis do host como `UV_INDEX_URL`
      mudam a origem da resolução sem entrar no hash das entradas. Em produção, a
      resolução passa por um proxy controlado (AWS CodeArtifact);
    - fixar a base por digest também congela as correções de segurança do Debian. O
      Dependabot não enxerga esse digest, porque ele está no `config.py`, não num
      Dockerfile; o processo de atualização (Renovate ou job agendado que abre PR,
      validado pelo Trivy) fica em "Com mais tempo";
    - o lock é resolvido para `--python-version 3.12`, e não para a versão exata
      (3.12.x) da imagem base; para os pacotes atuais, o resultado é o mesmo;
    - se o código do runtime mudar entre o cálculo do hash e o build dos wheels, a
      imagem ficaria com o rótulo errado; risco baixo, aceito;
  - **resolvido no E5:** os limites mínimos do runtime eram as versões mais recentes,
    herdadas do `uv add` (ex.: `pandas>=3.1.0`). Foram relaxados para as menores
    versões testadas (`numpy>=1.26.4`, `pandas>=2.2.2`), e um job do CI roda os testes
    do runtime com elas (ADR-021);
  - **achado (E2):** com o disco do host cheio, o Docker Desktop concluiu um build "com
    sucesso" e deixou arquivos truncados na imagem (`typing_extensions.py` sem
    `Sentinel`); o treino falhava com `ImportError` ou `SIGBUS`. É o motivo do
    self-check. Verificado num ambiente com os pacotes do lock: o self-check passa no
    ambiente íntegro e acusa o arquivo truncado. Custa alguns segundos, só em build
    novo.

### ADR-006 — `mlflow-skinny` no runtime e modelo salvo offline

- **Contexto:** o container de treino não tem rede nem credenciais, mas o modelo
  precisa sair num formato que o serving carregue.
- **Decisão:** o runner salva o modelo como MLflow pyfunc, offline, em `/output`. Ao
  receber o `/output`, o **host calcula os hashes por conta própria** (não confia nos
  hashes escritos pelo container) e só então registra no MLflow.
- **Alternativas descartadas:** serializar com `cloudpickle` e converter no host: o host
  precisaria desserializar um artefato não confiável, o que quebra a fronteira de confiança.
- **Consequências:**
  - assinatura e formato padrão do MLflow, sem reimplementação;
  - **hash garante integridade, não confiança.** O `train()` do cientista roda no mesmo
    processo que o runner e poderia alterar modelo e manifesto juntos. O hash calculado
    pelo host protege contra adulteração **depois** do treino (disco, registry,
    trânsito). O modelo em si continua não confiável: o risco ao carregá-lo depende do
    formato (ADR-017) e é contido pelo isolamento do container de serving;
  - **achado (E0):** o `mlflow-skinny` 3.17.2 não é tão enxuto. Puxa `databricks-sdk`,
    `gitpython`, `opentelemetry-*`, `protobuf` 7 e `sqlparse`, e o `mlship-runtime`
    resolve para 45 pacotes. Projetos que fixem `protobuf` antigo (comum com
    TensorFlow) vão conflitar; a co-resolução do ADR-005 detecta isso antes do treino;
  - **verificado (E1), com MLflow 3.17.2 e sem rede:**
    - o pyfunc exige `numpy` e `pandas`, que o `mlflow-skinny` não declara; sem eles,
      nem o `import mlflow.pyfunc` funciona. Os dois entram nas dependências do runtime;
    - modelos scikit-learn são salvos por padrão em `skops`, que exige o pacote
      `skops` no runtime (ver ADR-017). O `skops` depende do scikit-learn, que por isso
      entra em todo ambiente;
    - salvar offline, carregar pelo pyfunc e prever funciona, com predições idênticas
      às do modelo original;
  - ao ser importado, o MLflow escreve no log uma sugestão de carregar um "skill" de
    tracing. Os containers definem `MLFLOW_DISABLE_AGENT_HINT=1` para manter os logs limpos.

### ADR-007 — Imagem de serving por versão de modelo

- **Contexto:** o serving precisa do modelo; a questão é como ele chega à instância.
- **Decisão:** a imagem de serving é `FROM <imagem do lock>` + `COPY` do modelo
  verificado. O build é feito pelo `serve`, com cache por versão, mantendo o `promote`
  como operação pura de registry.
- **Alternativas descartadas:** baixar o modelo do registry no startup: exigiria
  credenciais e acesso ao MLflow no container que recebe tráfego externo, e o startup
  passaria a depender do registry estar no ar.
- **Consequências:** serving sem credenciais; artefato imutável (ambiente + modelo num
  digest); rollback é subir a imagem anterior; trocar o `champion` exige rebuild e
  redeploy. Não escala bem para modelos de gigabytes: nesse caso, a alternativa
  descartada volta a ser a melhor, e o desenho AWS mostra as duas.
- **Champion e challenger:** cada alias vira uma imagem e uma instância próprias
  (`serve --alias champion` e `serve --alias challenger`). A divisão de tráfego fica
  fora da API, num roteador ou load balancer; nunca dois modelos no mesmo container.
  Ver ADR-013.

### ADR-008 — MLflow local com SQLite

- **Decisão:**
  - servidor MLflow via `compose.yaml`, com backend SQLite e artefatos em volume
    nomeado. É o padrão do CLI (`MLSHIP_TRACKING_URI=http://127.0.0.1:5000`);
  - o servidor faz **proxy dos artefatos** (`--artifacts-destination`): o CLI envia o
    modelo pelo servidor e nunca recebe credencial do armazenamento. Trocar o volume
    pelo S3 muda só a configuração do servidor, como no MLflow gerenciado do SageMaker;
  - os testes unitários usam o mesmo código com SQLite direto (sem servidor). Os mesmos
    testes rodam contra um servidor com `MLSHIP_TEST_TRACKING_URI`.
- **Alternativas descartadas:**
  - o mesmo com MinIO imitando o S3: cerca de meio dia de trabalho por pouco ganho na
    demonstração;
  - SQLite direto como padrão, sem servidor: zero infraestrutura, mas o cliente
    gravaria os artefatos no destino; em produção, precisaria de credencial do S3 em
    cada máquina que treina;
  - backend em arquivo (`./mlruns`): no MLflow 3.17 está em "maintenance mode" e o
    cliente recusa usá-lo sem uma variável de opt-out.
- **Consequências:**
  - setup de um comando (`docker compose up -d`), com healthcheck;
  - o servidor não tem autenticação: só escuta em `127.0.0.1`. Em produção, o MLflow
    gerenciado do SageMaker usa IAM;
  - o `mlflow-skinny` não traz `sqlalchemy` nem `alembic`; os dois entram só como
    dependências de desenvolvimento, para os testes com SQLite;
  - **verificado (E3):** os testes dos repositórios passam com SQLite e contra um
    `mlflow server` 3.17.2 real por HTTP, com artefatos indo para o destino do servidor;
  - a imagem `ghcr.io/mlflow/mlflow:v3.17.2` acompanha a versão do cliente no lock e é
    fixada pelo digest do índice multiarquitetura (`sha256:2cc40b32…2fc0`, consultado
    em 24/10); **verificado:** sobe no Docker Desktop e o healthcheck fica `Healthy`.

### ADR-009 — Docker SDK para Python

- **Decisão:** o adapter de containers usa o Docker SDK.
- **Alternativas descartadas:** `subprocess` com lista de argumentos: sem dependência
  extra e transparente, mas sem tipos e mais difícil de testar.
- **Consequências:**
  - o cliente Docker é criado sob demanda, no primeiro uso, e não na montagem dos
    serviços. Assim, comandos que não usam containers (`validate`, `runs`, `promote`)
    e os testes unitários rodam sem um daemon Docker ativo;
  - o cliente é injetável, e o mapeamento de erros (Docker parado, imagem ausente,
    build com falha) é testado com um fake, sem Docker;
  - o SDK constrói imagens pelo builder clássico, não pelo BuildKit. Funciona com o
    Docker 29 (verificado, inclusive com o containerd image store do Docker Desktop),
    mas é o caminho que o Docker está descontinuando. Alternativa futura: chamar
    `docker buildx build` pela linha de comando.

### ADR-010 — Dados do exemplo verificados por hash

- **Contexto:**
  - o container de treino não tem rede, então os dados precisam existir antes;
  - o `pandas.to_csv` usa a quebra de linha do sistema operacional por padrão:
    confirmado com CRLF no Windows, o que daria hashes diferentes para os mesmos
    dados em cada sistema;
  - nem a [página de origem](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
    nem a descrição do scikit-learn trazem declaração de licença.
- **Decisão:**
  - `scripts/download_california_housing.py`, com dependências em lock próprio, baixa no host;
  - gera o CSV com `lineterminator="\n"`;
  - compara o SHA-256 em memória com o valor esperado e só grava o arquivo se bater;
  - os dados não são redistribuídos no repositório.
- **Alternativas descartadas:**
  - versionar o CSV no git: seria redistribuir um dado sem licença declarada; dados
    ficam no histórico do git para sempre; e o repositório deve guardar a referência
    (script + hash), não o dado (ver ADR-014). O risco aceito é depender da fonte,
    mitigado com cache no CI (E6) e dados sintéticos nos testes unitários;
  - documentar o hash sem verificar: não detectaria mudança na fonte.
- **Consequências:**
  - o hash fica público, como os do `uv.lock`: é de integridade, não de sigilo;
  - o risco real é alguém **alterar** o hash esperado, e isso se controla com revisão
    no repositório (no desenho de produção: hashes de referência no registry, com IAM);
  - atualizar a fonte ou as versões exige atualizar o hash de propósito.

### ADR-011 — Limites de reprodutibilidade

- **Contexto:** o mesmo `train()` com `seed=42` produziu métricas e predições idênticas
  em duas execuções no mesmo ambiente. Algoritmos com paralelismo (o
  `HistGradientBoostingRegressor` usa OpenMP) podem variar no último dígito entre
  sistemas operacionais ou números de threads.
- **Decisão:**
  - o runner fixa as seeds globais;
  - o runner fixa o número de threads (`OMP_NUM_THREADS` e equivalentes iguais ao
    limite de CPU do container) e grava no manifesto: o paralelismo depende das CPUs da
    máquina, não da imagem;
  - **mesma imagem e mesma configuração de threads**: os testes exigem predições idênticas;
  - **entre ambientes**, a comparação usa tolerância (`numpy.allclose`);
  - não se compara o manifesto inteiro nem os bytes do modelo serializado.
- **Consequências:** a promessa documentada é reprodutibilidade de entradas (código,
  dados, lock, imagem) e de resultados dentro da tolerância, não determinismo bit a bit.

### ADR-012 — Versões das ferramentas fixadas

- **Decisão:**
  - `uv` fixado em `==0.13.5` (`[tool.uv] required-version`); o `setup-uv` do CI lê o
    mesmo campo, então há uma única fonte;
  - backend de build `uv_build` na faixa 0.13;
  - hooks do pre-commit rodam as versões de ruff e mypy do `uv.lock`, sem uma segunda
    lista de versões;
  - actions do GitHub fixadas por SHA (tags podem ser movidas);
  - `.gitattributes` com `eol=lf`, para o mesmo conteúdo em Windows e Linux.
- **Alternativas descartadas:** não fixar o `uv`: versões diferentes podem regravar o
  `uv.lock` de formas diferentes, e o CI usaria a mais recente.
- **Consequências:** atualizar o `uv` passa a ser uma mudança deliberada em um único lugar.

### ADR-013 — Champion e challenger com gate de promoção

- **Contexto:**
  - mover o alias `champion` sem comparação permite que um modelo pior entre em
    produção sem ninguém perceber;
  - comparar métricas de modelos avaliados com dados ou splits diferentes engana: a
    diferença pode vir dos dados, não do modelo.
- **Decisão:**
  - **offline (código, P2 do E3):** `promote --alias champion` compara o candidato com o
    champion atual e só promove se a métrica de comparação não piorar;
  - a comparação só acontece entre runs **comparáveis**: mesma métrica e direção
    declaradas, mesmo hash de dados e mesma seed no manifesto. Caso contrário, o
    `promote` recusa comparar e explica o motivo;
  - `--force` permite promover mesmo assim, deixando registrado no registry quem
    forçou e por quê. Localmente, "quem" é o usuário do sistema, sem autenticação; em
    produção, o principal do IAM;
  - a métrica e a direção são **declaradas no treino**, nos campos opcionais
    `primary_metric` e `direction` do `TrainOutput` (ADR-002), e gravadas no manifesto
    de cada run. No exemplo: `rmse`, `minimize`;
  - sem declaração, o gate não roda e o `promote` avisa que promoveu sem comparação;
  - o alias `challenger` é livre, sem gate: serve para colocar um candidato ao lado do
    champion;
  - **online (desenho):** canary e shadow com divisão de tráfego nos endpoints do
    SageMaker (variantes de produção com pesos e variantes shadow), cada variante sendo
    uma imagem por versão (ADR-007).
- **Alternativas descartadas:**
  - só documentar no desenho: sem governança concreta no código;
  - roteador local para dividir tráfego: reimplementa o que a AWS entrega pronto;
  - dois modelos no mesmo container: mistura versões numa instância e complica logs
    e rollback;
  - métrica escolhida por flag no `promote`: permitiria escolher, depois de ver os
    resultados, a métrica que favorece o candidato;
  - arquivo de config do projeto: mais um arquivo no contrato;
  - inferir a direção pelo nome da métrica: frágil com métricas de nome próprio;
  - métrica padrão definida pela ferramenta (ex.: F1): a ferramenta é genérica, e a
    métrica certa depende do custo do erro em cada problema.
- **Consequências:**
  - toda promoção vira uma decisão rastreável, inclusive as forçadas;
  - **limitação:** as métricas são calculadas pelo próprio código do cientista. Mesmos
    dados e mesma seed não garantem a mesma metodologia de avaliação: o candidato pode
    mudar o split ou calcular a métrica em outro conjunto. A solução é a ferramenta
    avaliar champion e candidato no mesmo holdout fixo (em "Com mais tempo");
  - métrica offline melhor não garante desempenho melhor em produção; por isso o
    desenho inclui shadow e canary antes da troca definitiva;
  - **se não couber no prazo:** o gate fica neste ADR e em "Com mais tempo", e o
    `promote` move o alias sem comparação.

### ADR-014 — Versionamento de dados por referência

- **Contexto:** todo run precisa ser rastreável até a versão exata dos dados que usou.
  Versionar dados não é o mesmo que colocá-los no git: o git guarda código e
  referências; os bytes dos dados ficam num armazenamento próprio, com versões imutáveis.
- **Decisão:**
  - **código:** o `train` calcula o hash de cada arquivo do diretório de dados e grava
    no manifesto do run. Dois runs com os mesmos hashes usaram exatamente os mesmos dados;
  - **desenho de produção:** dados em S3 com versionamento ou em tabelas Delta/Iceberg
    (time travel); o manifesto guarda URI + versão + hash.
- **Alternativas descartadas:**
  - dados no git (ou Git LFS): mistura código e dados, cresce o histórico para sempre e,
    em repositório público, redistribui o dado;
  - snapshot por conteúdo feito pela ferramenta (`.mlship/data/<hash>/` local, S3 em
    produção): resolveria a lacuna abaixo, mas custa armazenamento e não serve para
    volumes grandes, onde o certo é referenciar a versão; fica em "Com mais tempo";
  - integração com DVC: padrão de mercado, mas mais uma ferramenta para o cientista
    aprender, contra o contrato mínimo.
- **Consequências:**
  - todo run identifica a versão exata dos dados, e mudanças são detectadas;
  - **lacuna aceita:** localmente, versões antigas dos dados não ficam guardadas. Se
    os dados mudarem, sabemos qual hash um run antigo usou, mas não temos mais o
    arquivo para reproduzi-lo. Em produção, o armazenamento versionado fecha essa lacuna;
  - o dataset do exemplo é um caso particular (público, estático, uma versão): a
    referência dele é o hash esperado no script de download (ADR-010).

### ADR-015 — Versionamento de ponta a ponta e de modelos

- **Contexto:** em produção, toda predição deve ser rastreável até o código, os dados, o
  ambiente e os parâmetros que geraram o modelo. Isso exige que cada artefato do ciclo
  de vida tenha um identificador imutável e que exista um registro único ligando esses
  identificadores entre si.
- **Decisão:** o **manifesto de cada run** é o elo. Ele registra os identificadores de
  tudo o que entrou no treino e é vinculado ao run no MLflow; a versão do modelo aponta
  para esse run.

  | O quê | Mecanismo | Identificador | Onde fica | ADR |
  | --- | --- | --- | --- | --- |
  | Ferramenta | Semantic Versioning + commit | versão + git sha | manifesto | 015 |
  | Código do cientista | hash de cada arquivo do projeto (menos ocultos, caches, saídas e dados) | SHA-256 | manifesto | 015, 019 |
  | Dependências | lock co-resolvido com hashes (`uv`) | hash do lock | manifesto; tag da imagem | 005 |
  | Imagem de treino | tag = hash de todas as entradas do build; imutável | tag + digest | Docker local; ECR em produção | 005 |
  | Dados | hash por arquivo | SHA-256 por arquivo | manifesto; S3 versionado ou Delta em produção | 010, 014 |
  | Parâmetros e métricas | MLflow tracking | `run_id` | MLflow | 006 |
  | Resultado da validação | gravado no run | `run_id` | manifesto | 015 |
  | Modelo | MLflow Model Registry | nome + versão (v1, v2…) + SHA-256 | MLflow | 006, 015 |
  | Alias | ponteiro móvel para uma versão | `champion`, `challenger` | MLflow | 013, 015 |
  | Imagem de serving | uma por versão de modelo | tag por versão + digest | Docker local; ECR em produção | 007 |
  | Config e infraestrutura | git | commit | repositório | 012 |
  | Ferramentas de build e CI | versões fixadas | `uv==0.13.5`; actions por SHA | `pyproject.toml`; workflow | 012 |

  **Modelos, em particular:**
  - cada `train` bem-sucedido registra uma nova versão. O número é sequencial,
    atribuído pelo registry, e **imutável**: a v3 é sempre o mesmo artefato;
  - o consumo é sempre por **alias explícito**, nunca por "latest". `promote` só move o
    ponteiro;
  - o `serve` resolve o alias **uma vez**, no build da imagem. A instância responde e
    registra nos logs o **número da versão**, nunca o nome do alias;
  - o hash do modelo é verificado antes de qualquer carregamento;
  - **rollback** é mover o alias para a versão anterior e subir a imagem dela, que já
    existe no cache.
- **Alternativas descartadas:**
  - consumir "a última versão": o que está em produção mudaria a cada treino, sem decisão;
  - stages do MLflow (Staging, Production): marcados como obsoletos a partir do MLflow
    2.9, em favor de aliases, tags e modelos registrados separados
    ([RFC](https://github.com/mlflow/mlflow/issues/10336));
  - Semantic Versioning manual para modelos: alguém precisaria decidir o que é "major"
    num modelo, e o número não diria nada que o manifesto não diga melhor;
  - tag do git como versão do modelo: o mesmo código gera modelos diferentes com dados
    ou parâmetros diferentes.
- **Consequências:**
  - a partir da versão servida chega-se ao run, e do run a código, dados, ambiente e
    parâmetros;
  - o nome do modelo registrado é o da pasta do projeto, ou `--model-name` (ADR-019);
  - implementação: E3 (registro, aliases; ADR-019) e E4 (serving por versão).

### ADR-016 — Observabilidade do serving e referência para drift

- **Contexto:** o ciclo de vida de um modelo não termina no deploy: é preciso ver como
  ele se comporta em produção e ter uma referência para detectar degradação. O
  serving já expõe `/metrics` (Prometheus) e a versão do modelo em cada resposta e
  log, mas sem um painel esses dados não chegam a quem opera.
- **Decisão:**
  - **dashboard (P1, E7):** Prometheus e Grafana no `compose.yaml`, com dashboard
    provisionado e versionado no repositório: latência p50/p95/p99, throughput, taxa de
    erro e versão do modelo servida. O `serve` informa o endereço do dashboard;
  - **referência para drift (P2, E2):** o runner grava no manifesto estatísticas das
    features de treino (média, quantis). É a base que qualquer detecção de drift
    precisa;
  - **desenho de produção:** CloudWatch e SageMaker Model Monitor; alarmes de
    degradação disparam um pipeline de retreino, que passa pelo gate de promoção
    (ADR-013) antes do deploy. É esse loop que fecha o ciclo de vida.
- **Alternativas descartadas:**
  - só documentar o `/metrics`: os dados existem, mas ninguém os vê sem um `curl`;
  - comando `mlship monitor` com relatório de drift: alto valor, mas não cabe no prazo;
    um comando novo no fluxo do cientista só se justifica com esse relatório;
  - retreino automático implementado: no desenho, com serviços gerenciados.
- **Consequências:**
  - na apresentação, o teste de carga e o dashboard mostram a latência ao vivo;
  - o drift fica preparado (referência no manifesto e predições amostradas), mas não
    implementado;
  - **se não couber no prazo:** fica o `/metrics` e este ADR, e o dashboard vai para
    "Com mais tempo".

### ADR-017 — Formatos de modelo por nível de confiança

- **Contexto:**
  - o MLflow 3.17 salva modelos scikit-learn em `skops`, que desserializa sem executar
    código. Mas, por padrão, o `skops` bloqueia tipos considerados arriscados, entre eles
    os de quase todo modelo de árvore. Testado: `Ridge` salva; `DecisionTreeRegressor`,
    `RandomForestRegressor` e `GradientBoostingRegressor` bloqueiam
    `sklearn.tree._tree.Tree`; o `HistGradientBoostingRegressor` do exemplo bloqueia o
    `TreePredictor`. Num arquivo adulterado, esses tipos podem fazer o `predict()`
    derrubar o processo ou ler memória fora dos limites;
  - formatos como `cloudpickle` aceitam qualquer objeto, mas carregar o modelo executa
    código arbitrário.
- **Decisão:** um caminho seguro por padrão e uma saída explícita, sem restringir o
  tipo de modelo:
  - **nível 1, seguro (padrão):** estimadores e pipelines scikit-learn em `skops`, com
    uma allowlist revisada (os tipos de árvore acima). A plataforma pode ampliá-la por
    configuração (`MLSHIP_SKOPS_TRUSTED_TYPES`). Tipo fora da lista falha o treino com
    mensagem que diz qual tipo é e como liberá-lo;
  - **nível 2, com aceitação explícita:** qualquer objeto com `.predict()`, via pyfunc
    genérico em `cloudpickle`, só com `train --allow-pickle`. A escolha fica no
    manifesto, a CLI e o `serve` avisam, e a plataforma pode desligar essa opção por
    configuração;
  - **nível 3 (desenho, "Com mais tempo"):** formatos nativos de XGBoost, LightGBM ou
    ONNX, sem pickle. Ainda a verificar se os flavors do MLflow usam esses formatos;
  - a confiança é concedida **por quem opera** (configuração do host ou flag), nunca
    pelo `train.py`; os tipos confiáveis e o nível usado vão para o manifesto.
- **Alternativas descartadas:**
  - restringir a v1 a scikit-learn: seguro, mas fecha a plataforma para modelos comuns
    e empurra usuários para fora dela;
  - o cientista declarar os tipos confiáveis no `TrainOutput`: quem é avaliado
    concederia a própria confiança;
  - `cloudpickle` para todos os modelos: carregar passaria a executar código arbitrário
    por padrão;
  - confiar em todos os tipos que o `skops` reportar: anula o controle, e a própria
    biblioteca desaconselha.
- **Consequências:**
  - o exemplo funciona sem flags; modelos customizados também funcionam, com o risco
    aceito e registrado;
  - a lista de tipos confiáveis fica gravada no `MLmodel`, que o hash calculado pelo
    host protege contra adulteração (ADR-006);
  - o risco residual (travamento ou leitura de memória nos tipos de árvore; execução
    de código no nível 2) é contido pelo container de serving: sem credenciais, sem rede
    de saída, sistema de arquivos read-only, usuário não-root.

### ADR-018 — Treino isolado e contrato com o runner

- **Contexto:** o `train.py` é código não confiável e roda com os dados do cientista.
  Tudo o que sai do container também é não confiável. A plataforma precisa conter o
  que o código pode fazer e decidir, ela mesma, o que aceitar da saída.
- **Decisão:**
  - **antes do container:** validação do projeto (sintaxe e assinatura do `train` via
    AST; `requirements.txt` sem `--index-url`, `-e`, `git+`, URLs ou caminhos locais;
    diretório de dados existente e com arquivos). Erro bloqueia antes do lock; todos os
    erros aparecem de uma vez. As regras completas estão no ADR-021;
  - **container:** `--network none`; rootfs read-only; `/tmp` em tmpfs (`nosuid,nodev`);
    projeto, dados e `run.json` montados read-only; só `/output` gravável;
    `--cap-drop ALL`; `no-new-privileges`; limites de CPU, memória (sem swap), processos
    e tempo; perfil seccomp padrão do Docker. Nenhuma credencial, nenhuma variável do
    host e nenhum socket do Docker entram: só as variáveis que a ferramenta define
    (`PYTHONHASHSEED`, threads, `HOME=/tmp`, caminho do `run.json`);
  - **usuário:** sempre não-root. Em Linux e macOS, o uid de quem roda a ferramenta (as
    saídas pertencem a ele); no Windows ou com host root, o usuário 10001 da imagem;
  - **contrato:** entrada `run.json` (parâmetros, seed, política de serialização),
    saída `result.json` (status, todos os erros com código, mensagem e dica, métricas,
    formato do modelo), `metrics.json` e `model/`. Os dois JSON têm `schema_version`;
  - **o host não confia na saída:** o `train.py` roda no mesmo processo que o runner,
    então pode reescrever o `result.json`. Por isso o arquivo é lido com limite de
    tamanho, schema conferido e textos truncados, e nada é executado. Links simbólicos
    são recusados (`result.json` e `MLmodel`), para que um link para um arquivo do host
    não seja lido como saída. Timeout e falta de memória prevalecem sobre o que o
    arquivo diz; sucesso exige código de saída 0 e modelo presente;
  - **falhas legíveis:** processo morto por sinal vira `TRAIN_CRASHED` com o nome do
    sinal (`SIGBUS`, `SIGSEGV`, `SIGKILL`…) e uma dica, e o erro avisa quando não houve
    nenhuma saída de log. A tabela de sinais é a do Linux, fixa no código, porque o
    host pode ser Windows. Recusa do Docker em criar o container (montagem não
    compartilhada, por exemplo) vira erro com dica, sem stack trace; Ctrl+C remove o
    container e sai com código 130. Falha ao salvar o modelo vira `MODEL_SAVE_FAILED`,
    erro do projeto, e não "bug do mlship";
  - **saídas por run:** `.mlship/runs/<run_id>/`, com `run.json` e `output/`. É a área
    de trânsito antes do registro no MLflow (E3);
  - **seed:** `seed` padrão 42, sempre repassada ao `train()` em `params`, para que o
    código veja exatamente a seed usada.
- **Alternativas descartadas:**
  - configuração por variáveis de ambiente: tudo vira texto e aparece em
    `docker inspect`; o JSON montado é o mesmo arquivo que vai para o manifesto;
  - erros pelo código de saída e logs: só diria "falhou"; ler logs estruturados seria
    frágil, porque o código do cientista escreve no mesmo stream;
  - saídas numa pasta global ou temporária: misturaria projetos, ou perderia a saída
    se o MLflow estivesse fora do ar;
  - sandbox de kernel (gVisor, Firecracker): isolamento mais forte que containers, que
    compartilham o kernel do host; fica no desenho de produção e em "Com mais tempo".
- **Consequências:**
  - **verificado (E2), com daemon Docker real e uma sonda estática dentro do
    container:** sem rede; rootfs, projeto e dados read-only; `/tmp` gravável; usuário
    não-root; só as variáveis definidas pela ferramenta; logs em tempo real; timeout
    mata o container em cerca de 3 s; nenhum container sobra; e, via `docker inspect`,
    `cap_drop ALL`, `no-new-privileges`, memória sem swap, limites de CPU e processos;
  - os mesmos comportamentos, com a imagem Python real, são cobertos por testes de
    integração (treino do exemplo, sonda de isolamento em Python, timeout) que rodam no
    CI (Linux) e passaram também no Windows com Docker Desktop (24/10), junto com o
    self-check da imagem;
  - sem rede, os dados precisam estar no disco antes do treino (ADR-010);
  - `/tmp` não usa `noexec`, porque algumas bibliotecas executam arquivos temporários;
  - o timeout é verificado a cada 0,5 s;
  - **feito no E3 (ADR-019):** ao registrar o `model/` no MLflow, o host percorre o
    diretório sem seguir links e recusa qualquer coisa que não seja arquivo ou
    diretório comum, além de limitar tamanho e número de arquivos. Senão, um link
    criado pelo código do cientista poderia levar um arquivo do host para o artifact
    store.

### ADR-019 — Lineage, registro e promoção

- **Contexto:** o ADR-015 define o que versionar; falta decidir como cada `train` vira
  um registro rastreável, inclusive quando falha, e como uma versão chega a um alias.
- **Decisão:**
  - **um ID de ponta a ponta:** o run do MLflow é aberto antes de qualquer etapa cara
    (lock, build, treino), com status `RUNNING`. O ID dele nomeia
    `.mlship/runs/<run_id>/`. O run termina como `FINISHED`, `FAILED` (com o erro numa
    tag) ou `KILLED` (Ctrl+C). Projetos inválidos na validação estática não abrem run;
  - **fail-fast:** antes do run, uma chamada ao MLflow sem retentativas e com timeout
    de 5 s. Servidor fora do ar é avisado antes de qualquer build ou treino;
  - **um experimento por modelo:** o nome do modelo é o da pasta do projeto
    (normalizado), ou `--model-name` (`MLSHIP_MODEL_NAME`). Experimento e modelo
    registrado têm o mesmo nome;
  - **manifesto (`manifest.json`, `schema_version` 1), calculado no host:**
    - ferramenta: versão, commit e `git_dirty` (mudanças não commitadas nos pacotes da
      ferramenta);
    - código: SHA-256 de **cada arquivo do projeto**, porque um `config.yaml` lido pelo
      `train.py` muda o modelo tanto quanto o código. Ficam de fora arquivos e pastas
      ocultos (`.env`, `.git`: nem o nome é gravado), `.mlship/`, caches, virtualenvs,
      a pasta de dados quando está dentro do projeto e o `mlship.lock` (registrado em
      dependências). Arquivos acima de 1 MB são listados com o tamanho, sem hash;
    - dependências: hash do lock e das entradas do lock;
    - ambiente: tag e ID da imagem, chave do ambiente e as entradas que a compõem;
    - dados: local (relativo ao projeto quando está dentro dele, para não gravar o
      caminho do usuário) e SHA-256 de cada arquivo;
    - params, seed, limites de recursos e métricas;
    - modelo: SHA-256 de cada arquivo e um hash único do conjunto (hash do mapa
      ordenado `{caminho: sha256}`);
    - erros e avisos da validação e do treino;
  - **saída não confiável:** o diretório do modelo é percorrido **sem seguir links**.
    Link simbólico, FIFO ou dispositivo faz o run falhar (`MODEL_UNTRUSTED_FILES`) e
    nada é enviado (pendência do ADR-018). Tamanho e número de arquivos são conferidos
    pelos metadados antes de ler um byte (padrão 2 GB e 10.000 arquivos,
    `MLSHIP_MODEL_MAX_BYTES` e `MLSHIP_MODEL_MAX_FILES`; acima disso, `MODEL_TOO_LARGE`);
  - **o que o MLflow recusaria é recusado antes:** o MLflow só valida nomes e tamanhos
    ao registrar, no fim do run, e a regra muda com o sistema do cliente (`:` vale no
    Linux e não no Windows). A ferramenta aplica a regra mais restrita no início:
    - params: nome válido, valor até 6.000 caracteres, até 50 params, e nenhum nome com
      cara de segredo (`password`, `token`, `api_key`…): params ficam gravados para
      sempre, e o container sem rede não usaria credencial nenhuma;
    - métricas: nome válido e até 100, checados no runner (erro do projeto, com dica)
      e de novo no host;
    - a regra existe no runner e na CLI, que não podem importar um ao outro; um teste
      de contrato garante que as duas dizem a mesma coisa;
  - **registro:** params, métricas e tags com os hashes principais (para busca na UI);
    artefatos `model/`, `manifest.json`, `run.json` e `result.json`. Run com sucesso
    cria uma nova versão do modelo, com o hash do modelo numa tag da versão; o run
    recebe o número da versão;
  - **promoção:** `mlship promote <versão> --alias champion` só move o ponteiro e grava
    na versão quem promoveu (usuário do sistema), quando e qual versão foi substituída.
    `latest` é reservado e recusado; promover a versão que já tem o alias não muda nada.
    O run também recebe a tag padrão `mlflow.user` (quem treinou), que a interface do
    MLflow mostra na coluna "User";
  - **`mlship runs`:** os runs do modelo, do mais novo ao mais antigo, com status,
    versão e métricas;
  - **o run nunca fica `RUNNING` por engano:** qualquer exceção, inclusive Ctrl+C
    durante o upload, fecha o run como `FAILED` ou `KILLED`. Se o registro falha depois
    do treino, modelo e manifesto ficam no disco e a mensagem diz onde;
  - **credenciais e privacidade:** usuário e senha numa URI (ex.:
    `MLSHIP_TRACKING_URI=https://user:senha@host`) são mascarados em toda mensagem de
    erro. A telemetria de uso que o MLflow 3 envia por padrão fica desligada na CLI, na
    imagem e no servidor do compose (`MLFLOW_DISABLE_TELEMETRY`); quem quiser liga.
- **Alternativas descartadas:**
  - abrir o run só depois do sucesso: dois IDs (o local e o do MLflow) e falhas
    invisíveis no tracking;
  - um experimento global: mistura projetos; filtrar por tag é frágil;
  - commit da ferramenta gravado no build do pacote: correto para distribuição, mas
    exige um passo de build que o projeto ainda não tem; vai para o desenho de
    produção;
  - hash do modelo calculado pelo runner: viria de dentro da fronteira não confiável
    (ADR-006);
  - stages do MLflow (Staging/Production): obsoletos, ver ADR-015.
- **Consequências:**
  - da versão servida chega-se ao run, e do run ao manifesto: código, dados,
    dependências, ambiente, parâmetros e ferramenta;
  - um run feito com a ferramenta modificada e não commitada fica marcado
    (`git_dirty=true`): é rastreável, mas não reproduzível a partir do commit;
  - hashear os dados lê todos os arquivos mais uma vez a cada `train`; aceitável aqui,
    em produção o manifesto referencia a versão no S3/Delta (ADR-014);
  - a promoção ainda não compara métricas; o gate do ADR-013 fica para depois do E4;
  - **limites conhecidos, registrados:**
    - a integridade do modelo se ancora no hash guardado numa tag da versão; quem tem
      escrita no registry pode trocar artefato e tag juntos. Em produção: bucket com
      S3 Object Lock, IAM restrito ao pipeline e assinatura dos artefatos ("Fora de
      escopo");
    - "quem promoveu" é o usuário do sistema operacional, declarado pelo próprio
      ambiente (`getpass`); a auditoria confiável vem do principal do IAM em produção;
    - pastas de projetos diferentes com o mesmo nome caem no mesmo modelo registrado;
      `--model-name` resolve;
    - a cota de disco durante o treino não existe localmente (bind mount sem limite); o
      limite do modelo protege o host depois do treino. Em produção, cada job tem seu
      próprio volume com tamanho fixo;
  - **verificado (E3):** serviço de treino de ponta a ponta contra um servidor MLflow
    3.17.2 real (run, artefatos, versão com hash, tag da versão no run); `runs` e
    `promote` pela CLI, incluindo versão inexistente, alias reservado e servidor fora
    do ar; nomes de métrica testados contra o validador do próprio MLflow
    (`"rmse (test)"` e `"f1@0.5"` recusados); senha na URI não aparece na saída; sem
    tentativas de conexão de telemetria depois do opt-out. O treino real em container
    com registro roda no job de integração do CI.

### ADR-020 — Serving da versão promovida

- **Contexto:** servir é onde o modelo encontra tráfego externo. O que chega ao
  container tem de ser exatamente o que foi registrado, no ambiente em que foi
  treinado, carregado só com a confiança que a plataforma concede, atrás de uma API
  com contrato explícito e observável.
- **Decisão:**
  - **o alias é resolvido uma vez, no host:** `mlship serve --alias champion` resolve a
    versão e, a partir daí, só o número da versão existe. A instância responde e
    registra a versão em cada resposta, log e métrica (`mlship_model_info`);
  - **âncora de confiança: o banco do tracking, não o artifact store.** A versão do
    registry e as tags do run (no banco do MLflow) dizem o que deve chegar; nada que
    venha do artifact store (S3 em produção) é usado antes de bater com elas. Quem só
    tem escrita no bucket não consegue trocar o modelo, o lock nem a política de carga.
    O `manifest.json` e o `run.json` do artifact store são registro para humanos e não
    entram nessa decisão;
  - **nada é usado antes de verificado:** o modelo é baixado do registry a cada
    `serve` (nunca de um cache local), percorrido sem seguir links, com limite de
    tamanho, e o SHA-256 tem de bater com o registrado na versão. Dentro do container,
    o processo verifica de novo antes de carregar;
  - **mesma imagem do treino (D1):** o run guarda o `mlship.lock` como artefato e, nas
    tags, o hash do lock e a chave do ambiente. Se a imagem existe, é reutilizada; se
    foi apagada, é reconstruída a partir daquele lock **só se o hash do lock bater com
    a tag e a chave recalculada for idêntica**. Lock diferente é adulteração; chave
    diferente (a ferramenta mudou) é erro com explicação, nunca um ambiente "parecido";
  - **imagem de serving por versão (ADR-007):** `FROM <imagem do treino>` + modelo
    verificado + identidade (nome, versão, hash) e política de carga em variáveis de
    ambiente. A tag inclui versão, hash do modelo, imagem base e política: muda
    qualquer um, outra imagem;
  - **política de carga decidida pela plataforma (ADR-017):** o `MLmodel` foi escrito
    no mesmo processo que o código do cientista, então é dado, não decisão. Antes de
    desserializar, o servidor lê o descritor com `yaml.safe_load` (via MLflow) e só
    aceita o caminho seguro (flavor `sklearn` em skops, `python_function` delegando a
    `mlflow.sklearn`, sem código embutido, tipos dentro da allowlist da plataforma).
    Qualquer outra coisa só é aceita se a tag do run registrar o modelo como salvo em
    cloudpickle, o que só acontece com `--allow-pickle` no treino. Um modelo em skops é
    servido sem permissão de pickle, mesmo que o run a tivesse;
  - **API (`/v1/predict`):**
    - schema gerado da assinatura do modelo (D2): o OpenAPI mostra as colunas reais;
      tipos estritos (`"3"` não vira número, `true` não vira 1); `NaN` e `Infinity`
      (aceitos pelo parser JSON do Python) são 422, não uma falha do modelo; coluna
      desconhecida é erro; todos os erros de campo de uma vez, com a posição
      (`rows.3.MedInc`), sem
      ecoar os valores enviados;
    - uma rota para 1 a N linhas (D3), predição vetorizada numa chamada; limite de
      linhas (padrão 1.000) e de bytes (padrão 1 MB, recusado com 413 antes do parse).
      O corpo é lido até o limite antes de chegar à aplicação, o que vale também para
      corpo em chunks, sem `Content-Length`;
    - predição fora do event loop (thread), com timeout (504);
    - erros sempre em JSON com código, mensagem e `request_id`, nunca stack trace. O
      erro inesperado vira 500 no middleware mais externo, para que entre nas
      métricas, no log de acesso e no `X-Request-ID` como qualquer outra resposta
      (o handler de exceção do Starlette roda fora dos middlewares e deixaria o 500
      invisível para um alerta de taxa de erro);
    - `X-Request-ID` aceito se seguro, senão gerado; devolvido em toda resposta;
    - API key opcional (`MLSHIP_SERVING_API_KEY`), comparada em tempo constante;
  - **operação:** `/health/live` e `/health/ready` separados; `/metrics` (Prometheus)
    na mesma porta e sem a API key (D5): latência por rota em histograma (p50/p95/p99
    via `histogram_quantile`), requisições por status, linhas preditas e versão
    servida. Com vários workers, o modo multiprocesso do Prometheus agrega os
    processos. Logs em JSON, um por requisição, e amostragem opcional de entradas e
    predições (sem colunas excluídas por configuração) como base para drift (ADR-016);
  - **container:** o mesmo endurecimento do treino (não-root, rootfs read-only, `/tmp`
    em tmpfs, `cap_drop ALL`, `no-new-privileges`, limites de CPU, memória e
    processos). A diferença é a rede: a API precisa receber requisições, publicada só
    em `127.0.0.1`;
  - **`serve` em primeiro plano (D4):** mostra o que verificou, o endereço e os logs;
    Ctrl+C para e remove o container. Se a API parar sozinha (falta de memória,
    crash), o comando termina com erro, código de saída e dica. Quem quiser em segundo
    plano roda a imagem de serving com `docker run -d` (README).
- **Alternativas descartadas:**
  - baixar o modelo no startup do container: credenciais e dependência do registry
    no container que recebe tráfego (ADR-007);
  - reconstruir o ambiente "como der" quando a imagem sumiu: reabriria o skew de
    ambiente que o projeto evita;
  - lista genérica de registros validada só pelo MLflow: OpenAPI inútil e erro só no
    primeiro campo;
  - rotas separadas para lote: duas rotas com a mesma lógica. Inferência em lote de
    verdade (milhões de linhas) é um job offline, no desenho;
  - porta separada para métricas: em produção, quem separa é a rede (ALB roteando só
    `/v1/*`; `/metrics` e health só na VPC);
  - confiar nos `skops_trusted_types` gravados no `MLmodel`: o próprio código do
    cientista poderia ampliá-los;
  - ler a chave do ambiente e a permissão de pickle do `manifest.json` e do `run.json`
    baixados do artifact store (a primeira versão da E4): quem tivesse escrita só no
    bucket trocaria o lock por um com pacotes maliciosos e recalcularia a chave;
  - assinatura criptográfica dos artefatos (cosign, KMS): protege também contra quem
    escreve no banco, mas fica no desenho (fora de escopo).
- **Consequências:**
  - da resposta HTTP chega-se à versão, da versão ao run, e do run a tudo o que o
    gerou (ADR-015, ADR-019);
  - trocar o `champion` exige um novo `serve` (nova imagem, novo container): a
    instância nunca muda de modelo sozinha. Rollback é servir a versão anterior, cuja
    imagem já existe;
  - cada worker carrega o modelo na sua memória: memória ≈ workers × modelo;
  - o timeout responde 504, mas a thread da predição termina sozinha; uma predição
    patológica ainda consome CPU até acabar (limite de CPU do container);
  - modelos treinados antes desta versão (sem as tags de ambiente) precisam ser
    treinados de novo para serem servidos;
  - **limites conhecidos, registrados:** quem tem escrita no banco do MLflow consegue
    mudar a versão e as tags juntas; em produção, isso é IAM restrito ao pipeline de
    CI/CD, e o desenho prevê assinatura. O container de serving tem saída para a
    rede (o Docker não separa entrada de saída sem orquestrador); em produção, security
    groups sem egress. A API key local é uma variável de ambiente do container; em
    produção, IAM ou JWT no API Gateway e segredos no Secrets Manager;
  - **verificado (E4):** API com um modelo real salvo pelo runner (contrato, erros,
    limites, auth, timeout, métricas, request id); política de carga contra `MLmodel`
    adulterado (tipos fora da allowlist, loader trocado, código embutido, pickle sem
    permissão); hash adulterado impede o carregamento; o entrypoint real com 2 workers
    e métricas agregadas; o fluxo do host com fakes (alias, verificação, imagem
    reutilizada ou reconstruída, chave divergente). O ponta a ponta com Docker
    (train → promote → serve → predict) roda no job de integração do CI.

### ADR-021 — Validação do projeto, lock e `init`

- **Contexto:** o cientista só fornece código. Quanto mais cedo um erro aparece, mais
  barato ele é: um pin incompatível descoberto depois de minutos de build, ou um
  `requests.get` que só falha dentro do container sem rede, são a pior experiência.
  A validação também é uma fronteira de segurança: o `requirements.txt` é a porta de
  entrada da supply chain. E nada disso pode executar o código do cientista.
- **Decisão:**
  - **dois níveis, tudo de uma vez:** erro viola o contrato e bloqueia; aviso é
    provável problema e não bloqueia; `--strict` (no `validate` e no `train`)
    transforma aviso em erro, para o CI. Cada problema traz código estável, local
    (`arquivo:linha`) e como corrigir;
  - **só leitura e sem executar:** o código é lido com `ast`, nunca importado, e a
    partir dos bytes, como o Python lê: BOM e `# coding:` valem; UTF-16 (o `>` do
    Windows PowerShell 5) é erro com a dica de como salvar em UTF-8. Pasta de projeto
    inexistente é um erro claro, não um traceback;
  - **só o que roda no container (R3):** `train.py` e os módulos do projeto que ele
    importa, seguidos transitivamente (`import utils`, `from features import x`,
    imports relativos dentro de pacotes), como o runner os encontra com a pasta do
    projeto no `sys.path`. Testes, scripts e notebooks que o `train.py` não importa
    ficam fora; sem isso, o `pytest` de `tests/` ou o `requests` de um script de
    download tornariam o `--strict` inútil no CI. Pastas ocultas e a pasta de dados
    ficam fora; erro de sintaxe num módulo importado é aviso (o Python só o acusa
    quando o import roda);
  - **`requirements.txt` (V1):** parser PEP 508 de referência (`packaging`), com
    continuação de linha e comentários. Erros:
    - opções de índice, `-e`, `-r`, `--hash` inline, URLs, VCS e caminhos locais;
    - sintaxe inválida, com a mensagem do parser;
    - pacotes da própria plataforma (`mlship-sdk`, `mlship-runtime`);
    - **conflito com o runtime (V2, estático):** pin exato fora da faixa que o runtime
      exige (`pandas==2.0.0` contra `pandas>=2.2.2`), com a faixa na mensagem.

    Aviso: versão sem pin exato (faixa ou curinga);
  - **resolução real (V2):** o `validate` resolve o `requirements.txt` com o runtime
    usando o mesmo `uv --no-build` do lock, numa pasta temporária: pega pacote
    inexistente e conflito transitivo, sem escrever no projeto e sem executar sdist.
    Um `mlship.lock` íntegro e atual já responde, sem rede. `--offline` pula a etapa;
    ela também é pulada quando o `requirements.txt` já tem erros. Índice inacessível
    vira um erro próprio (`REQUIREMENTS_INDEX_UNREACHABLE`), para não culpar o arquivo;
  - **lock commitado e atual (Y3):** o `validate` avisa quando o `mlship.lock` falta
    (`LOCK_MISSING`), não bate com o `requirements.txt` ou o runtime (`LOCK_STALE`) ou
    foi editado à mão (`LOCK_MODIFIED`). Com `--strict`, o CI exige um lock commitado
    e atual, como `uv lock --check` e `npm ci`. O comando `mlship lock` grava o lock
    sem treinar, depois de validar as dependências; o `train` não emite esses avisos,
    porque ele mesmo atualiza o lock;
  - **import sem pacote declarado (V3, aviso):** imports de todos os `.py`, menos
    stdlib, módulos locais, imports opcionais (`try/except ImportError`,
    `if TYPE_CHECKING`) e o que o runtime garante. Um mapa curado cobre os nomes que
    divergem (`sklearn` → `scikit-learn`, `cv2` → `opencv-python`); fora dele, nome
    igual. Mesma abordagem de deptry e pipreqs;
  - **código hostil ao container (V6, avisos):** cliente de rede (import ou URL
    `http(s)://`, `s3://`, `gs://`), caminho absoluto (`/home/…`, `C:\…`, `/dbfs/…`),
    variável de ambiente com cara de segredo, e trabalho pesado no nível do módulo
    (`read_csv`, `fit`, `open`, laços) fora de `train()`. São heurísticas: apontam o
    provável erro e não bloqueiam sozinhas;
  - **dados (V5):** pasta existe e tem arquivos (ocultos não contam); formato
    desconhecido é um aviso agregado com exemplos, porque quem lê os dados é o
    `train.py`;
  - **saída (V8):** texto para pessoas e `--format json` para CI, com os mesmos
    campos e `schema_version` para quem consome. Saída `0` válido, `1` bloqueado, `2`
    erro da ferramenta. O próximo comando sugerido inclui a pasta do projeto;
  - **`init` (V7):** template mínimo (`train.py` com scikit-learn, `requirements.txt`
    fixado, `README.md`, `.gitignore` e `data/example.csv` sintético) e o
    `mlship.lock` já gerado. Sem acesso ao índice, o projeto é criado assim mesmo e a
    saída indica o `mlship lock`. O projeto criado passa no `validate --strict` e
    treina como está (testado). Arquivos
    existentes nunca são sobrescritos sem `--force`, todos listados de uma vez; nada é
    escrito através de link simbólico;
  - **limites do runtime (V4):** mínimos relaxados para as menores versões testadas;
    o job `runtime-lowest` do CI instala o runtime com `--resolution lowest-direct` e
    roda os testes do runtime.
- **Alternativas descartadas:**
  - regex para o `requirements.txt` (como era no E2): aceita linhas inválidas que só
    quebram no `uv`, longe da linha do erro;
  - só a resolução do `uv`, sem regras estáticas: mensagem genérica, um erro por vez
    e dependência de rede para um erro que se vê no texto;
  - só regras estáticas: não pegam pacote inexistente nem conflito transitivo;
  - mapear imports pelos metadados dos wheels resolvidos (`top_level.txt`): exato,
    mas exige resolver e ler metadados de pacotes de terceiros;
  - rede como erro: falso positivo em import morto ou fallback; o container já
    bloqueia de fato, e o `--strict` cobre o CI;
  - formato de dados desconhecido como erro: limitaria o cientista sem motivo;
  - `init` interativo ou copiando o exemplo California Housing: pior no CI, ou
    específico demais e dependente de download;
  - varrer todos os `.py` do projeto (a primeira versão do E5) ou excluir pastas por
    convenção (`tests/`, `scripts/`): o primeiro gera falso positivo em todo projeto
    real; o segundo erra em qualquer layout fora da convenção;
  - SARIF para anotações no PR: fica em "Com mais tempo".
- **Consequências:**
  - o fluxo curto `init` → `validate` → `train` funciona numa pasta vazia, sem
    download;
  - o mesmo validador roda no `train`: o que passa no `validate --strict` do CI passa
    no `train --strict`;
  - as heurísticas podem errar para os dois lados; os códigos são estáveis e
    documentados, e nenhum aviso bloqueia sem `--strict`;
  - o mapa de imports é curado: um pacote cujo nome de import difere e não está no
    mapa gera aviso falso, corrigível com uma linha no mapa;
  - **limites conhecidos, registrados:** o conflito estático só compara pins exatos;
    faixas incompatíveis ficam para a resolução real. A resolução usa o índice
    configurado no `uv` do host (em produção, o proxy de pacotes). Imports dinâmicos
    (`importlib.import_module("x")`) não entram no grafo. Pacotes de namespace como
    `google.cloud` recebem a dica genérica `'google'`;
  - **verificado (E5):** um projeto com vários erros e avisos devolve todos; cada regra
    com casos positivos e negativos; resolução real com `uv` (pacote inexistente e
    projeto válido) sem escrever no projeto; `init` (com lock) → `validate --strict
    --offline` → treino do `train.py` gerado, também no Windows; arquivos em UTF-16,
    com BOM e latin-1 declarado; `tests/` e `scripts/` fora do grafo; lock ausente,
    desatualizado e editado; testes do runtime com `numpy` 1.26.4 e `pandas` 2.2.2.

### ADR-022 — Estratégia de testes e CI

- **Contexto:** o enunciado avalia a escolha do que testar, não a cobertura. A
  promessa da ferramenta é: o código do cientista roda isolado, o resultado é
  reproduzível e rastreável, e a API serve exatamente o modelo registrado. Os testes
  existem para provar essas garantias, e o CI, para que elas não regridam.
- **Decisão:**
  - **pirâmide em três níveis:**
    - **unitários** (cerca de 300, segundos, sem rede nem Docker): regras de
      validação, lock, manifesto, lineage contra um MLflow real em SQLite, API com um
      modelo real salvo pelo runner, política de carga, adapters com o Docker SDK
      falsificado, regras de arquitetura entre camadas;
    - **integração** (Docker e MLflow reais): resolução e build do ambiente, treino no
      container, isolamento de fato, timeout, serving ponta a ponta e
      reprodutibilidade;
    - **ponta a ponta pela CLI** (T1): o comando `mlship` instalado, como o cientista o
      usa, numa pasta vazia: `init → validate --strict → train → runs → promote →
      serve → predict → parar`. Usa os dados sintéticos do template, sem download, e
      verifica os códigos de saída e que o container some ao parar;
  - **reprodutibilidade (T2):** o mesmo projeto treinado duas vezes tem os mesmos
    hashes de entrada (código, dados, lock, ambiente), as mesmas métricas e **as mesmas
    predições sobre uma entrada fixa**. Não se comparam o manifesto inteiro (run id,
    datas) nem os bytes do modelo: a serialização não é determinística byte a byte, e o
    que importa é o comportamento (ADR-011);
  - **segurança no CI:**
    - **gitleaks (T3)** no histórico inteiro (um segredo apagado depois continua
      público) e no pre-commit; binário de versão fixa com checksum verificado, sem
      action de terceiros nem licença;
    - **Trivy (T4)** nas dependências travadas (`uv.lock`) e nas imagens de ambiente que
      os testes construíram. Bloqueia CRITICAL e HIGH **com correção disponível**: CVE
      sem patch na imagem base não tem ação possível e viraria ruído;
    - **Dependabot (T5)** para `uv` e GitHub Actions, semanal e agrupado;
  - **sem cache de build no CI (T6):** o job de integração levou 3m44s no último CI. O
    cache `type=gha` exige construir com `buildx`, e o mlship constrói pelo Docker
    SDK: seria testar outro caminho de build ou mudar o produto por causa do CI. E o
    build frio é o que pega falhas reais (imagem base nova, a imagem truncada do
    ADR-005). Revisitar se o job passar de uns 10 minutos, com cache nos PRs e build
    frio no `main`;
  - **parar o `serve` por sinal:** SIGTERM (`docker stop`, runners de CI, systemd) e
    Ctrl+Break no Windows param como Ctrl+C, removendo o container. Achado ao
    escrever o ponta a ponta: antes, só o Ctrl+C limpava;
  - **saída em UTF-8 mesmo redirecionada:** no Windows, pipe e arquivo usam a página de
    código ANSI (cp1252), que não tem as setas do mlship nem os emojis que o MLflow
    imprime no fim do run. `mlship train > log.txt`, ou qualquer log de CI no Windows,
    quebrava com `UnicodeEncodeError`. A CLI passa a escrever UTF-8 nesses casos. Achado
    pelo ponta a ponta rodado no Windows;
  - **testes não sujam o MLflow compartilhado:** integração e ponta a ponta apagam no
    fim os modelos e experimentos que criaram (o servidor do compose é o mesmo da
    demonstração).
- **Alternativas descartadas:**
  - ponta a ponta só pelos serviços (como era): não prova que os comandos funcionam
    juntos nem testa a saída que o cientista vê;
  - script shell no CI para o ponta a ponta: não roda no Windows e falha de forma
    menos legível;
  - comparar o hash do modelo salvo: falharia por motivo irrelevante;
  - `gitleaks-action`: exige licença em repositório de organização e dá acesso ao
    repositório a mais uma action;
  - Trivy só reportando: relatório que ninguém lê;
  - meta de cobertura: incentiva testar o fácil, não o que quebra a promessa.
- **Consequências:**
  - o CI tem quatro jobs: `lint-and-unit`, `runtime-lowest` (ADR-021), `integration`
    (com o ponta a ponta e o Trivy) e `secrets`;
  - uma imagem base com CVE corrigível passa a quebrar o CI até o digest ser
    atualizado: é o sinal desejado, e o processo de atualização está em "Com mais
    tempo". Um risco aceito vai para o `.trivyignore`, com justificativa;
  - **o que não é testado, e por quê:**
    - Docker no Windows e no macOS no CI: os runners hospedados não rodam containers
      Linux nesses sistemas; os testes de integração e ponta a ponta foram rodados no
      Windows à mão (registrado em cada ADR) e o mypy roda para as três plataformas;
    - AWS de verdade: o desenho de produção não tem implementação (fora de escopo);
    - carga e latência: ficam no E7 (Locust), não na suíte;
    - drift e retreino: não implementados (ADR-016);
    - pacote que só publica sdist: não é suportado por decisão (só wheels, ADR-005);
      a mensagem nesse caso é a do `uv`, não testada com um pacote real;
    - heurísticas de validação contra projetos reais variados: cobertas por casos
      escolhidos, não por um corpus;
    - reprodutibilidade entre máquinas ou com a imagem reconstruída: o teste treina
      duas vezes na mesma máquina e imagem. Entre hardwares diferentes, o ADR-011 já
      registra que a igualdade bit a bit não é garantida;
  - o Trivy no `uv.lock` inclui as ferramentas de desenvolvimento (ruff, mypy): elas
    rodam no CI com o código do repositório, então também são superfície de supply
    chain;
  - **a observar:** o `uv` fixado em `required-version` pode fazer o Dependabot recusar
    o lock se ele usar outra versão; se acontecer, a faixa vira `>=0.13.5,<0.14`.

---

## Fora de escopo

Itens fora do escopo do código; todos aparecem no desenho de arquitetura:

- multi-tenant e autenticação completa (local: API key opcional; no desenho: IAM/JWT);
- GPU e treino distribuído;
- busca de hiperparâmetros;
- monitoramento de drift implementado (base: log amostrado de predições);
- inferência em lote implementada;
- canary e shadow deploy (no desenho; o gate offline está no ADR-013);
- feature store implementada;
- assinatura criptográfica de artefatos;
- sandbox de kernel (gVisor, Firecracker).

## Com mais tempo

- Snapshot dos dados por conteúdo a cada `train` (`.mlship/data/<hash>/` local, S3 em
  produção), para tornar reproduzíveis runs antigos mesmo após mudança nos dados (ADR-014).
- Formatos nativos sem pickle para XGBoost, LightGBM e ONNX (nível 3 do ADR-017).
- Docker rootless e perfil seccomp próprio para o container de treino; em produção,
  sandbox de kernel (gVisor ou Firecracker) (ADR-018).
- Estágio final da imagem em distroless ou Chainguard (ADR-005).
- Atualização automática do digest da imagem base (Renovate ou job agendado que abre
  PR), com o Trivy validando a nova imagem (ADR-005).
- Resolver o lock para a versão exata do Python da imagem base (ADR-005).
- Commit da ferramenta gravado no build do pacote, em vez de lido do git na execução
  (ADR-019).
- Comando para registrar de novo um run cujo upload falhou (`mlship register <run>`),
  a partir do que ficou no disco (ADR-019).
- Saída SARIF no `validate`, para anotar os problemas direto no PR do GitHub (ADR-021).
- Avaliação feita pela ferramenta: champion e candidato avaliados no mesmo holdout fixo,
  para que o gate de promoção não dependa de métricas calculadas pelo código do
  cientista (ADR-013).

## Tempo gasto

| Dia | Data | Horas | Foco |
| --- | --- | --- | --- |
| D1 | 21/10 | | Planejamento, decisões A–D |
| D2 | 22/10 | | E0: estrutura, tooling, exemplo, CI |
| D3 | 23/10 | | E1 (lock e imagem) e E2 (treino isolado), revisão |
| D4 | 24/10 | | E3 (lineage e registro), revisão, testes no Windows, E4 (serving), revisão, E5 (validação e `init`) |
