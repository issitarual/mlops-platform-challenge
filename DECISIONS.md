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
| [005](#adr-005--um-lock-uma-imagem-para-treino-e-serving) | Um lock, uma imagem, para treino e serving | Aceita |
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

**Decisões previstas**, registradas quando implementadas:

| Tema | Épico |
| --- | --- |
| Isolamento do container de treino (sem rede, não-root, rootfs read-only, limites) | E2 |
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
  que muda o comportamento dela.
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

### ADR-005 — Um lock, uma imagem, para treino e serving

- **Contexto:** diferença de ambiente entre treino e serving é uma das causas clássicas
  de skew.
- **Decisão:** a ferramenta resolve o `requirements.txt` do cientista junto com o
  runtime num lock com hashes (`uv`), para Python e plataforma fixos (linux/amd64). A
  imagem recebe a tag igual ao hash do lock, com cache, e é a mesma no treino e no serving.
- **Alternativas descartadas:** instalar dependências em cada execução (lento e não
  reprodutível); uma imagem por projeto sem lock (versões flutuam entre builds).
- **Consequências:** conflitos com o runtime aparecem na resolução, antes do treino.
  Versões não fixadas pelo cientista viram aviso, não erro: o lock fixa e informa.

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
    trânsito). O modelo em si continua não confiável (carregar um pyfunc executa
    pickle): esse risco é contido pelo isolamento do container de serving;
  - **achado (E0):** o `mlflow-skinny` 3.17.2 não é tão enxuto. Puxa `databricks-sdk`,
    `gitpython`, `opentelemetry-*`, `protobuf` 7 e `sqlparse`, e o `mlship-runtime`
    resolve para 45 pacotes. Projetos que fixem `protobuf` antigo (comum com
    TensorFlow) vão conflitar; a co-resolução do ADR-005 detecta isso antes do treino;
  - **a verificar (E1):** se salvar e carregar como pyfunc exige `pandas`/`numpy` do
    lado do cientista.

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
- **Consequências:** o cliente Docker é criado sob demanda, no primeiro uso, e não na
  montagem dos serviços. Assim, comandos que não usam containers (`validate`, `runs`,
  `promote`) e os testes unitários rodam sem um daemon Docker ativo.

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
  | Imagem de treino | tag = hash do lock; imutável | tag + digest | Docker local; ECR em produção | 005 |
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
- Avaliação feita pela ferramenta: champion e candidato avaliados no mesmo holdout fixo,
  para que o gate de promoção não dependa de métricas calculadas pelo código do
  cientista (ADR-013).

## Tempo gasto

| Dia | Data | Horas | Foco |
| --- | --- | --- | --- |
| D1 | 21/10 | | Planejamento, decisões A–D |
| D2 | 22/10 | | E0: estrutura, tooling, exemplo, CI |
