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

**Decisões previstas**, registradas quando implementadas:

| Tema | Épico |
| --- | --- |
| Design da API de serving (schema pela assinatura, lote, erros, limites, health) | E4 |
| Validação do projeto em dois níveis (erro e aviso), estática via AST | E5 |

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
    - até a validação do E5, opções de supply chain no `requirements.txt` do cientista
      (`--index-url`, `-e`, `git+`) são aceitas pelo `uv` na resolução. O `train`
      valida o projeto antes de resolver o lock;
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
  - **ponto em aberto:** as dependências do runtime têm hoje limites mínimos iguais às
    versões mais recentes (ex.: `pandas>=3.1.0`), herdados do `uv add`. Isso aumenta a
    chance de conflito com projetos que fixam versões mais antigas. Rever junto com a
    validação de conflitos (E5);
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

- **Decisão:** servidor MLflow via `compose.yaml`, com backend SQLite e artefatos em
  volume local; URI de artefatos configurável para S3.
- **Alternativas descartadas:** o mesmo com MinIO imitando o S3: cerca de meio dia de
  trabalho por pouco ganho na demonstração.
- **Consequências:** setup de um comando. A imagem `ghcr.io/mlflow/mlflow:v3.17.2`
  acompanha a versão do cliente no lock; ainda falta confirmar a tag e fixar por digest (E3).

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
  | Código do cientista | hash de `train.py`, `requirements.txt` e demais `.py` do projeto | SHA-256 | manifesto | 015 |
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
  - o nome do modelo registrado por projeto (hoje `MLSHIP_MODEL_NAME`, com padrão
    `mlship-model`) é definido no E3;
  - implementação: E3 (registro, aliases) e E4 (serving por versão). Hoje existem só o
    tipo `ModelVersion` e a interface `ModelRegistry`.

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
    erros aparecem de uma vez. É o núcleo do E5, antecipado;
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
    CI e localmente;
  - sem rede, os dados precisam estar no disco antes do treino (ADR-010);
  - `/tmp` não usa `noexec`, porque algumas bibliotecas executam arquivos temporários;
  - o timeout é verificado a cada 0,5 s;
  - **obrigatório no E3:** ao registrar o `model/` no MLflow, o host percorre o
    diretório sem seguir links e recusa qualquer coisa que não seja arquivo ou
    diretório comum. Senão, um link criado pelo código do cientista poderia levar um
    arquivo do host para o artifact store.

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
- Avaliação feita pela ferramenta: champion e candidato avaliados no mesmo holdout fixo,
  para que o gate de promoção não dependa de métricas calculadas pelo código do
  cientista (ADR-013).

## Tempo gasto

| Dia | Data | Horas | Foco |
| --- | --- | --- | --- |
| D1 | 21/10 | | Planejamento, decisões A–D |
| D2 | 22/10 | | E0: estrutura, tooling, exemplo, CI |
| D3 | 23/10 | | E1 (lock e imagem) e E2 (treino isolado), revisão |
