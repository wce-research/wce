FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WCE_HOST=0.0.0.0 \
    WCE_PORT=8000

WORKDIR /app
RUN groupadd --system wce && useradd --system --gid wce --home-dir /app wce
COPY --chown=wce:wce app/ app/
COPY --chown=wce:wce core/ core/
COPY --chown=wce:wce language/ language/
COPY --chown=wce:wce physics/ physics/
COPY --chown=wce:wce logic/ logic/
COPY --chown=wce:wce docs/hypotheses_generative_certificates.md docs/hypotheses_generative_certificates.md
COPY --chown=wce:wce docs/architecture_wce.md docs/architecture_wce.md
COPY --chown=wce:wce results/predictive_coverage.json results/continuum_summary.json results/active_summary.json results/
COPY --chown=wce:wce docs/identity/ docs/identity/
USER wce

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)" || exit 1
CMD ["python", "-m", "app.web"]
