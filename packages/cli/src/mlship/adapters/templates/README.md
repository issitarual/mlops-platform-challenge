Dockerfile templates rendered by the CLI for each user project. The data scientist
never writes a Dockerfile. Any change to a template changes the environment key
(ADR-005), so the next `train` builds a new image.
