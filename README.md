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
uv run pytest tests/unit
```

## Subir o MLflow

```bash
docker compose up -d      # tracking e registry em http://127.0.0.1:5000
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

## Estrutura

| Pacote | Onde roda | Papel |
| --- | --- | --- |
| `packages/sdk` (`mlship_sdk`) | código do cientista | contrato `TrainOutput`, sem dependências |
| `packages/runtime` (`mlship_runtime`) | dentro da imagem | runner de treino e API de serving |
| `packages/cli` (`mlship`) | máquina do usuário | CLI `init`, `validate`, `train`, `runs`, `promote`, `serve` |

Decisões e alternativas: [DECISIONS.md](DECISIONS.md). Uso de IA: [AI_USAGE.md](AI_USAGE.md).
