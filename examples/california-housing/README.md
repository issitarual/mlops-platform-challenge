# Exemplo: California Housing

Regressão do valor mediano de imóveis por quarteirão na Califórnia, com um
`Pipeline` scikit-learn (`StandardScaler` + `HistGradientBoostingRegressor`).

## Dados

Os dados não ficam no repositório. Para baixar, rode na raiz:

```bash
uv run scripts/download_california_housing.py
```

O script grava `examples/california-housing/data/california_housing.csv` (20.640
linhas, 8 features + alvo `MedHouseVal`) somente se o SHA-256 bater com o esperado. Se o arquivo
de origem mudar, o script falha e não grava nada.

- **Fonte:** repositório StatLib, via `sklearn.datasets.fetch_california_housing`.
  Descrição original: [Luís Torgo, Regression Datasets](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html).
  Dados derivados do Censo dos EUA de 1990.
- **Citação:** Pace, R. Kelley and Ronald Barry, *Sparse Spatial Autoregressions*,
  Statistics and Probability Letters, 33 (1997) 291-297.
- **Licença:** nem a página de origem nem a descrição do scikit-learn trazem uma
  declaração de licença. Por isso o repositório **não redistribui** os dados: eles
  são baixados da fonte por quem roda o exemplo.

## Parâmetros

| Parâmetro | Padrão |
| --- | --- |
| `seed` | 42 |
| `learning_rate` | 0.1 |
| `max_iter` | 200 |
