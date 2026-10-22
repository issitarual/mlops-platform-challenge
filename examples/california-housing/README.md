# Exemplo: California Housing

Regressão do valor mediano de imóveis por quarteirão na Califórnia, com um
`Pipeline` scikit-learn (`StandardScaler` + `HistGradientBoostingRegressor`).

## Dados

Os dados não ficam no repositório. Para baixar, rode na raiz:

```bash
uv run scripts/download_california_housing.py
```

O script grava `examples/california-housing/data/california_housing.csv` e
imprime o SHA-256 do arquivo.

- **Fonte:** dataset California Housing (StatLib; Pace & Barry, 1997), obtido via
  `sklearn.datasets.fetch_california_housing`.
- **Licença:** TODO — confirmar a licença na fonte original antes da entrega.

## Parâmetros

| Parâmetro | Padrão |
| --- | --- |
| `seed` | 42 |
| `learning_rate` | 0.1 |
| `max_iter` | 200 |
