"""Training runner executed inside the isolated container.

Plain modules on purpose (no controller/service/repository layers): it is a short,
linear, one-shot program. See DECISIONS.md (ADR-004).

Contract with the host (ADR-018):
- input: ``run.json`` (path in ``MLSHIP_RUN_CONFIG``), mounted read-only;
- output: ``result.json``, ``metrics.json`` and ``model/`` in the output directory,
  the only writable mount. The host treats all of it as untrusted data.
"""
