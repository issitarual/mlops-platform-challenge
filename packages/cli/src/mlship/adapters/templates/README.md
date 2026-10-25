Templates the CLI renders.

- `environment.Dockerfile`, `serving.Dockerfile`: one per user project or model
  version. The data scientist never writes a Dockerfile. Any change to the
  environment template changes the environment key (ADR-005), so the next `train`
  builds a new image.
- `project/`: what `mlship init` writes (ADR-021). `.tmpl` keeps the template code out
  of this package's modules and checks; `dot-` becomes a leading dot.
