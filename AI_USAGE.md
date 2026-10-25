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
  ficou registrada no `DECISIONS.md` (ADR-001 a ADR-020).
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

### Achados em que a IA ajudou a diagnosticar

- **Imagem corrompida com o disco cheio.** O treino morria com `SIGBUS` e sem log. A IA
  comparou o `typing_extensions.py` dentro da imagem com o arquivo correto e concluiu
  que o build terminou "com sucesso" gravando arquivos truncados. Daí veio o self-check
  da imagem (ADR-005) e as mensagens legíveis por sinal (ADR-018).
- **Telemetria do MLflow.** Durante os testes, o proxy do ambiente da IA registrou
  conexões recusadas para `config.mlflow-telemetry.io`: o MLflow 3 envia telemetria de
  uso por padrão. Ficou desligada na CLI, na imagem e no servidor (ADR-019).

## Limites do uso de IA

- O ambiente da IA não alcançava o Docker Hub nem o `ghcr.io`: o build real das
  imagens, o digest do servidor MLflow e os testes no Windows foram feitos por mim e
  pelo CI.
