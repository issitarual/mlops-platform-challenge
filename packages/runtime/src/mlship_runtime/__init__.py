"""Code that runs inside the image built for each user project.

Two independent entry points share the same image (and therefore the same
environment), which removes training/serving environment skew:

- ``mlship_runtime.runner``: one-shot training job, no network, no credentials.
- ``mlship_runtime.serving``: the REST API that serves one model version.
"""
