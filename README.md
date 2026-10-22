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

## Estrutura

| Pacote | Onde roda | Papel |
| --- | --- | --- |
| `packages/sdk` (`mlship_sdk`) | código do cientista | contrato `TrainOutput`, sem dependências |
| `packages/runtime` (`mlship_runtime`) | dentro da imagem | runner de treino e API de serving |
| `packages/cli` (`mlship`) | máquina do usuário | CLI `init`, `validate`, `train`, `runs`, `promote`, `serve` |

Decisões e alternativas: [DECISIONS.md](DECISIONS.md). Uso de IA: [AI_USAGE.md](AI_USAGE.md).
