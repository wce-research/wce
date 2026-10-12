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
COPY --chown=wce:wce docs/hypotheses_active_structure.md docs/hypotheses_active_structure.md
COPY --chown=wce:wce docs/hypotheses_generative_certificates.md docs/hypotheses_generative_certificates.md
COPY --chown=wce:wce docs/architecture_wce.md docs/architecture_wce.md
COPY --chown=wce:wce results/predictive_coverage.json results/continuum_summary.json results/active_summary.json results/
COPY --chown=wce:wce docs/rapport_robust_reconstruction.md docs/rapport_robust_reconstruction.md
COPY --chown=wce:wce docs/rapport_deep_candidate_generation.md docs/rapport_deep_candidate_generation.md
COPY --chown=wce:wce docs/rapport_primitive_discovery.md docs/rapport_primitive_discovery.md
COPY --chown=wce:wce docs/rapport_downstream_selection_loss.md docs/rapport_downstream_selection_loss.md
COPY --chown=wce:wce docs/rapport_validated_optimizer.md docs/rapport_validated_optimizer.md
COPY --chown=wce:wce docs/rapport_objective_alignment.md docs/rapport_objective_alignment.md
COPY --chown=wce:wce docs/rapport_robust_primitive_discovery.md docs/rapport_robust_primitive_discovery.md
COPY --chown=wce:wce docs/rapport_partial_observation_parameters.md docs/rapport_partial_observation_parameters.md
COPY --chown=wce:wce docs/rapport_fragment_type_inference.md docs/rapport_fragment_type_inference.md
COPY --chown=wce:wce docs/rapport_boundary_ownership.md docs/rapport_boundary_ownership.md
COPY --chown=wce:wce docs/rapport_deep_composition_oracle_atoms.md docs/rapport_deep_composition_oracle_atoms.md
COPY --chown=wce:wce docs/rapport_end_to_end_discovered_atoms.md docs/rapport_end_to_end_discovered_atoms.md
COPY --chown=wce:wce docs/rapport_latent_primitive_recovery.md docs/rapport_latent_primitive_recovery.md
COPY --chown=wce:wce docs/rapport_high_precision_primitives.md docs/rapport_high_precision_primitives.md
COPY --chown=wce:wce results/robust_summary.json results/robust_summary.json
COPY --chown=wce:wce results/deep_generation_summary.json results/deep_generation_summary.json
COPY --chown=wce:wce results/deep_primitive_summary.json results/deep_primitive_summary.json
COPY --chown=wce:wce results/deep_downstream_summary.json results/deep_downstream_summary.json
COPY --chown=wce:wce results/deep_validated_summary.json results/deep_validated_summary.json
COPY --chown=wce:wce results/deep_objective_summary.json results/deep_objective_summary.json
COPY --chown=wce:wce results/deep_robust_primitive_summary.json results/deep_robust_primitive_summary.json
COPY --chown=wce:wce results/deep_partial_observation_summary.json results/deep_partial_observation_summary.json
COPY --chown=wce:wce results/deep_fragment_type_summary.json results/deep_fragment_type_summary.json
COPY --chown=wce:wce results/deep_boundary_ownership_summary.json results/deep_boundary_ownership_summary.json
COPY --chown=wce:wce results/deep_composition_oracle_summary.json results/deep_composition_oracle_summary.json
COPY --chown=wce:wce results/deep_end_to_end_summary.json results/deep_end_to_end_summary.json
COPY --chown=wce:wce results/deep_latent_primitive_summary.json results/deep_latent_primitive_summary.json
COPY --chown=wce:wce results/deep_high_precision_summary.json results/deep_high_precision_summary.json
COPY --chown=wce:wce docs/hypotheses_deep_compositional_discovery.md docs/hypotheses_deep_compositional_discovery.md
COPY --chown=wce:wce docs/architecture_validated_transformation.md docs/architecture_validated_transformation.md
COPY --chown=wce:wce docs/sealed_sets.md docs/sealed_sets.md
COPY --chown=wce:wce docs/identity/ docs/identity/
USER wce

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)" || exit 1
CMD ["python", "-m", "app.web"]
