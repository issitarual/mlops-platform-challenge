# mlship

Ferramenta de MLOps que leva código Python de treino até uma API REST: treino
isolado e reprodutível em Docker, rastreamento no MLflow e serving com FastAPI.

> **Status:** em construção.

## Requisitos

- [uv](https://docs.astral.sh/uv/)
- Docker (a partir do treino)

## Desenvolvimento

```bash
uv sync --all-packages      # instala os 3 pacotes do workspace + ferramentas de dev
uv run mlship --help
uv run pytest tests/unit    # segundos, sem rede nem Docker
```

Com Docker e o MLflow no ar (`docker compose up -d`), os testes de integração e o
ponta a ponta pela CLI (`init` → `validate` → `train` → `promote` → `serve` → `predict`):

```bash
export MLSHIP_TRACKING_URI=http://127.0.0.1:5000      # PowerShell: $env:MLSHIP_TRACKING_URI = "http://127.0.0.1:5000"
uv run pytest -m "integration or e2e" tests/integration tests/e2e -v
```

O CI roda também os testes do runtime com as versões mínimas declaradas, o gitleaks no
histórico e o Trivy nas dependências e nas imagens (ADR-022).

## Subir o MLflow

```bash
docker compose up -d      # tracking e registry em http://127.0.0.1:5000
```

## Começar um projeto

```bash
uv run mlship init my-model                       # train.py, requirements.txt, lock e dados de exemplo
uv run mlship validate my-model --data my-model/data
uv run mlship train my-model --data my-model/data
```

O projeto criado passa na validação e treina como está; troque o modelo e os dados
pelos seus. O contrato é só `train.py` com `train(data_dir, params) -> TrainOutput` e
um `requirements.txt`.

O `validate` lê o código sem executá-lo (o `train.py` e os módulos que ele importa)
e mostra **todos** os problemas de uma vez, cada um com o local e como corrigir:
assinatura do `train`, opções proibidas e pins incompatíveis com o runtime no
`requirements.txt`, imports sem pacote declarado, acesso à rede, caminhos absolutos,
segredos lidos do ambiente, a pasta de dados e o `mlship.lock` (ausente ou
desatualizado). Ele também resolve as dependências com o runtime (`--offline` pula
essa etapa). Avisos não bloqueiam; com `--strict` viram erro.

Depois de mudar o `requirements.txt`, `uv run mlship lock my-model` atualiza o
`mlship.lock`; commite os dois. Para CI:

```bash
uv run mlship validate my-model --strict --format json   # saída 0 válido, 1 bloqueado
```

## Treinar o exemplo

Requer Docker em execução e alguns GB livres em disco (no Windows, no disco onde o
Docker Desktop guarda os dados).

```bash
uv run scripts/download_california_housing.py
uv run mlship train examples/california-housing --data examples/california-housing/data
```

Na primeira vez, a ferramenta resolve as dependências (`mlship.lock`) e constrói a
imagem do ambiente, que é verificada arquivo por arquivo antes do uso; nas seguintes,
reaproveita as duas. O treino roda num container sem
rede, read-only e com limites, e o resultado fica em
`examples/california-housing/.mlship/runs/<run_id>/`, com o `manifest.json` (hashes
de código, dados, dependências, ambiente e modelo). Parâmetros vão com `-p`, por
exemplo `-p seed=7 -p learning_rate=0.05`.

Cada treino com sucesso vira uma nova versão do modelo `california-housing` (nome da
pasta; `--model-name` muda). O MLflow não precisa estar aberto no navegador, mas
precisa estar no ar: a ferramenta avisa antes de construir ou treinar qualquer coisa.

## Acompanhar e promover

```bash
uv run mlship runs examples/california-housing                # runs, versões e métricas
uv run mlship promote 1 --project examples/california-housing # alias champion -> v1
```

O serving sempre segue um alias explícito (`champion`, `challenger`), nunca "a
última versão".

## Servir o modelo

```bash
uv run mlship serve --project examples/california-housing
```

O `serve` resolve o alias (padrão `champion`), baixa o modelo, confere o SHA-256 com o
registrado, monta a imagem de serving sobre a imagem exata do treino e sobe a API em
http://127.0.0.1:8080. Ctrl+C para e remove o container. Opções: `--alias`, `--port`,
`--workers`.

```bash
curl -s http://127.0.0.1:8080/v1/predict -H "content-type: application/json" -d '{
  "rows": [{"MedInc": 8.3, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02,
            "Population": 322.0, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}]
}'
```

A resposta traz as predições, o modelo e a versão (`{"model": {"name":
"california-housing", "version": "2"}, ...}`) e o `request_id`. A documentação da API
fica em `/docs`; health em `/health/live` e `/health/ready`; métricas em `/metrics`.

- **API key (opcional):** defina `MLSHIP_SERVING_API_KEY` antes do `serve`; as chamadas
  a `/v1/*` passam a exigir o header `X-API-Key`.
- **Em segundo plano:** a imagem de serving é autossuficiente:
  `docker run -d -p 127.0.0.1:8080:8080 <imagem mostrada pelo serve>`.

## Estrutura

| Pacote | Onde roda | Papel |
| --- | --- | --- |
| `packages/sdk` (`mlship_sdk`) | código do cientista | contrato `TrainOutput`, sem dependências |
| `packages/runtime` (`mlship_runtime`) | dentro da imagem | runner de treino e API de serving |
| `packages/cli` (`mlship`) | máquina do usuário | CLI `init`, `validate`, `train`, `runs`, `promote`, `serve` |

Decisões e alternativas: [DECISIONS.md](DECISIONS.md). Uso de IA: [AI_USAGE.md](AI_USAGE.md).
