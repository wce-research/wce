PYTHON := python3

.DEFAULT_GOAL := help

.PHONY: help test test-world check clean

help:
	@echo "WCE — World Configuration Engine"
	@echo ""
	@echo "Commandes disponibles :"
	@echo "  make test        Exécute tous les tests"
	@echo "  make test-world  Teste le modèle du monde"
	@echo "  make check       Vérifie la syntaxe Python"
	@echo "  make clean       Supprime les fichiers temporaires"
	@echo "  make docker-up   Construit et démarre le site sur le Dell"
	@echo "  make docker-down Arrête le site"
	@echo "  make docker-build Reconstruit l'image"
	@echo "  make docker-logs Suit les journaux"
	@echo "  make docker-dev  Démarre avec synchronisation des changements"

.PHONY: docker-up docker-down docker-build docker-logs docker-ps docker-dev

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-build:
	docker compose build --no-cache

docker-logs:
	docker compose logs -f web

docker-ps:
	docker compose ps

docker-dev:
	docker compose -f compose.yaml -f compose.dev.yaml up --build --watch

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

test-world:
	$(PYTHON) -m tests.test_world

check:
	$(PYTHON) -m compileall -q core physics language logic app tests
	@echo "Syntaxe OK"

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	@echo "Nettoyage terminé"

.PHONY: test-transformation

test-transformation:
	$(PYTHON) -m tests.test_transformation

.PHONY: test-history

test-history:
	$(PYTHON) -m tests.test_history

.PHONY: test-constraint

test-constraint:
	$(PYTHON) -m tests.test_constraint

.PHONY: test-gravity

test-gravity:
	$(PYTHON) -m tests.test_gravity

.PHONY: test-ground-collision

test-ground-collision:
	$(PYTHON) -m tests.test_ground_collision

.PHONY: test-lift-and-fall

test-lift-and-fall:
	$(PYTHON) -m tests.test_lift_and_fall

.PHONY: test-actions

test-actions:
	$(PYTHON) -m tests.test_lift_and_fall

.PHONY: test-language

test-language:
	$(PYTHON) -m tests.test_language


.PHONY: repl

repl:
	$(PYTHON) -m app.repl

.PHONY: inspect-tests

inspect-tests:
	@grep -n "def test_location_after_lift" tests/test_wce.py

.PHONY: inspect-wall-legacy

inspect-wall-legacy:
	@echo "=== Usages de 'side' dans le code ==="
	@grep -rn '"side"' --include="*.py" . | grep -v __pycache__ | grep -v ".pyc"
	@echo ""
	@echo "=== Moteur moderne (physics/interactions.py) ==="
	@grep -n '"side"' physics/interactions.py
	@echo ""
	@echo "=== Adaptateur legacy : 1 usage (ligne 42 dans _legacy_wall_to_collider_and_material) ==="
	@grep -n 'side = entity_properties.get' physics/interactions.py
	@echo ""
	@echo "=== Documentation / exemple : 1 usage ==="
	@grep -n '"side": "right"' physics/interactions.py | head -1
	@echo ""
	@echo "=== Tests legacy + tests de compatibilité ==="
	@grep -n '"side"' tests/test_wce.py | wc -l
	@echo "usages dans tests/test_wce.py"

.PHONY: demo-geometry

demo-geometry:
	$(PYTHON) scripts/demo_geometry.py

.PHONY: demo-transformations demo-reasoning-core demo-algebra demo-reasoning demo-planner demo-learning

demo-transformations:
	$(PYTHON) scripts/demo_transformations.py

demo-reasoning-core:
	$(PYTHON) scripts/demo_reasoning_core.py

demo-algebra:
	$(PYTHON) scripts/demo_algebra.py

demo-reasoning:
	$(PYTHON) scripts/demo_reasoning.py

demo-planner:
	$(PYTHON) scripts/demo_planner.py

demo-learning:
	$(PYTHON) scripts/demo_learning.py

.PHONY: demo-plasticity test-plasticity
demo-plasticity:
	$(PYTHON) scripts/demo_plasticity.py

test-plasticity:
	$(PYTHON) -m unittest tests.test_plasticity -v

.PHONY: demo-lifelong test-lifelong
demo-lifelong:
	$(PYTHON) scripts/demo_lifelong.py

test-lifelong:
	$(PYTHON) -m unittest tests.test_lifelong -v

.PHONY: demo-symbolic-lifelong demo-language-bridge test-symbolic
demo-symbolic-lifelong:
	$(PYTHON) scripts/demo_symbolic_lifelong.py

demo-language-bridge:
	$(PYTHON) scripts/demo_language_bridge.py

test-symbolic:
	$(PYTHON) -m unittest tests.test_symbolic -v

.PHONY: demo-language-core demo-hallucination-audit test-language-core
demo-language-core:
	$(PYTHON) scripts/demo_language_core.py

demo-hallucination-audit:
	$(PYTHON) scripts/demo_hallucination_audit.py

test-language-core:
	$(PYTHON) -m unittest tests.test_language_core -v

.PHONY: demo-independent-language demo-independent-hallucination test-independent
demo-independent-language:
	$(PYTHON) scripts/demo_independent_language.py

demo-independent-hallucination:
	$(PYTHON) scripts/demo_independent_hallucination.py

test-independent:
	$(PYTHON) -m unittest tests.test_independent -v

.PHONY: demo-grounded-language demo-coverage-growth test-grounded
demo-grounded-language:
	$(PYTHON) scripts/demo_grounded_language.py

demo-coverage-growth:
	$(PYTHON) scripts/demo_coverage_growth.py

test-grounded:
	$(PYTHON) -m unittest tests.test_grounded -v

.PHONY: demo-temporal-belief demo-predictive-uncertainty test-temporal temporal-multibench temporal-sealed-eval
demo-temporal-belief:
	$(PYTHON) scripts/demo_temporal_belief.py

demo-predictive-uncertainty:
	$(PYTHON) scripts/demo_predictive_uncertainty.py

test-temporal:
	$(PYTHON) -m unittest tests.test_temporal -v

temporal-multibench:
	$(PYTHON) -c "import json; from core.temporal_eval import multi_bench; json.dump(multi_bench(8), open('results/temporal_multibench.json','w'), indent=1, default=str)"

temporal-sealed-eval:
	$(PYTHON) scripts/temporal_sealed_eval.py

.PHONY: demo-source-calibration demo-safe-identity test-sources source-multiseed source-sealed-eval
demo-source-calibration:
	$(PYTHON) scripts/demo_source_calibration.py

demo-safe-identity:
	$(PYTHON) scripts/demo_safe_identity.py

test-sources:
	$(PYTHON) -m unittest tests.test_sources -v

source-multiseed:
	$(PYTHON) scripts/source_multiseed.py 16

source-sealed-eval:
	$(PYTHON) scripts/source_sealed_eval.py

.PHONY: demo-knowledge-boundary demo-lexical-calibration test-epistemic epistemic-multiseed sealed-blind-eval
demo-knowledge-boundary:
	$(PYTHON) scripts/demo_knowledge_boundary.py

demo-lexical-calibration:
	$(PYTHON) scripts/demo_lexical_calibration.py

test-epistemic:
	$(PYTHON) -m unittest tests.test_epistemic -v

epistemic-multiseed:
	$(PYTHON) scripts/epistemic_multiseed.py 8
	$(PYTHON) scripts/epistemic_multiseed.py 16 no_elimination,callex,entitled

sealed-blind-eval:
	$(PYTHON) scripts/sealed_blind_eval.py

.PHONY: demo-discourse-memory demo-active-learning test-discourse demo-unified-belief demo-third-party-discourse test-unified unified-multiseed
demo-unified-belief:
	$(PYTHON) scripts/demo_unified_belief.py

demo-third-party-discourse:
	$(PYTHON) scripts/demo_third_party_discourse.py

test-unified:
	$(PYTHON) -m unittest tests.test_unified -v

unified-multiseed:
	$(PYTHON) scripts/unified_multiseed.py 8

demo-discourse-memory:
	$(PYTHON) scripts/demo_discourse_memory.py

demo-active-learning:
	$(PYTHON) scripts/demo_active_learning.py

test-discourse:
	$(PYTHON) -m unittest tests.test_discourse -v

# ---- algèbre de mesure et d'observation
.PHONY: demo-measurement-algebra demo-representation-basis demo-scientific-agent test-measurement measurement-multiseed measurement-sealed-eval
demo-measurement-algebra:
	$(PYTHON) scripts/demo_measurement_algebra.py

demo-representation-basis:
	$(PYTHON) scripts/demo_representation_basis.py

demo-scientific-agent:
	$(PYTHON) scripts/demo_scientific_agent.py

test-measurement:
	$(PYTHON) -m unittest tests.test_measurement -v

measurement-multiseed:
	$(PYTHON) -u -c "import json; from core.measurement_agent import multi_seed; json.dump(multi_seed(range(16)), open('results/measurement_multiseed.json','w'), indent=1, default=str)"

measurement-sealed-eval:
	$(PYTHON) scripts/measurement_sealed_eval.py

# ---- perturbation, sensibilité et stabilité des invariants
.PHONY: demo-perturbation-stability demo-invariant-duality demo-safe-plasticity demo-generalized-credit test-perturbation perturbation-multiseed measurement-value-multiseed perturbation-sealed-eval
demo-perturbation-stability:
	$(PYTHON) scripts/demo_perturbation_stability.py

demo-invariant-duality:
	$(PYTHON) scripts/demo_invariant_duality.py

demo-safe-plasticity:
	$(PYTHON) scripts/demo_safe_plasticity.py

demo-generalized-credit:
	$(PYTHON) scripts/demo_generalized_credit.py

test-perturbation:
	$(PYTHON) -m unittest tests.test_perturbation -v

perturbation-multiseed:
	$(PYTHON) -u -c "import json; from core.perturbation_benchmark import multi_seed; json.dump(multi_seed(range(16)), open('results/perturbation_multiseed.json','w'), indent=1, default=str)"

measurement-value-multiseed:
	$(PYTHON) -u -c "import json; from core.measurement_value import multi_seed_policies; json.dump(multi_seed_policies(range(16)), open('results/measurement_value_multiseed.json','w'), indent=1, default=str)"

perturbation-sealed-eval:
	$(PYTHON) scripts/perturbation_sealed_eval.py

# ---- plasticité compositionnelle
.PHONY: demo-compositional-plasticity demo-learned-stabilizer demo-no-replay-learning demo-active-credit test-compositional compositional-multiseed compositional-sealed-eval
demo-compositional-plasticity:
	$(PYTHON) scripts/demo_compositional_plasticity.py

demo-learned-stabilizer:
	$(PYTHON) scripts/demo_learned_stabilizer.py

demo-no-replay-learning:
	$(PYTHON) scripts/demo_no_replay_learning.py

demo-active-credit:
	$(PYTHON) scripts/demo_active_credit.py

test-compositional:
	$(PYTHON) -m unittest tests.test_compositional -v

compositional-multiseed:
	$(PYTHON) -u -c "import json; from core.compositional_benchmark import multi_seed; json.dump(multi_seed(range(16)), open('results/compositional_multiseed.json','w'), indent=1, default=str)"

compositional-sealed-eval:
	$(PYTHON) scripts/compositional_sealed_eval.py

# ---- certificats génératifs, second ordre, monde continu
.PHONY: demo-generative-certificate demo-second-order-stability demo-capacity-pressure demo-skill-regeneration demo-law-discovery test-generative continuum-multiseed continuum-aggregate continuum-sealed-eval continuum-scaling
demo-generative-certificate:
	$(PYTHON) scripts/demo_generative_certificate.py

demo-second-order-stability:
	$(PYTHON) scripts/demo_second_order_stability.py

demo-capacity-pressure:
	$(PYTHON) scripts/demo_capacity_pressure.py

demo-skill-regeneration:
	$(PYTHON) scripts/demo_skill_regeneration.py

demo-law-discovery:
	$(PYTHON) scripts/demo_law_discovery.py

test-generative:
	$(PYTHON) -m unittest tests.test_generative -v

SEEDS ?= 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
continuum-multiseed:
	mkdir -p results/continuum
	for s in $(SEEDS); do $(PYTHON) scripts/continuum_seed.py $$s || exit 1; done
	$(PYTHON) scripts/continuum_aggregate.py

continuum-aggregate:
	$(PYTHON) scripts/continuum_aggregate.py

continuum-scaling: continuum-aggregate

continuum-sealed-eval:
	$(PYTHON) scripts/continuum_sealed_eval.py

# ---- induction de programmes de frontière
.PHONY: demo-boundary-induction demo-family-induction demo-active-program-discovery demo-truth-anchored-certificate demo-neural-crystallization test-boundary boundary-multiseed boundary-scaling boundary-sealed-eval
demo-boundary-induction:
	$(PYTHON) scripts/demo_boundary_induction.py

demo-family-induction:
	$(PYTHON) scripts/demo_family_induction.py

demo-active-program-discovery:
	$(PYTHON) scripts/demo_active_program_discovery.py

demo-truth-anchored-certificate:
	$(PYTHON) scripts/demo_truth_anchored_certificate.py

demo-neural-crystallization:
	$(PYTHON) scripts/demo_neural_crystallization.py

test-boundary:
	$(PYTHON) -m unittest tests.test_boundary -v

BSEEDS ?= 16
BPROCS ?= 12
boundary-multiseed:
	$(PYTHON) scripts/boundary_multiseed.py $(BSEEDS) $(BPROCS)

boundary-scaling: boundary-multiseed

boundary-sealed-eval:
	$(PYTHON) scripts/boundary_sealed_eval.py

# ---- découverte structurelle active
.PHONY: demo-active-structural-discovery demo-functional-equivalence demo-noise-aware-truth demo-depth-guided-search demo-functional-family-induction demo-hard-capacity test-active active-calibrate active-structure-multiseed active-structure-ablation active-structure-noise-sweep active-structure-scaling active-structure-sealed-eval
demo-active-structural-discovery:
	$(PYTHON) scripts/demo_active_structural_discovery.py

demo-functional-equivalence:
	$(PYTHON) scripts/demo_functional_equivalence.py

demo-noise-aware-truth:
	$(PYTHON) scripts/demo_noise_aware_truth.py

demo-depth-guided-search:
	$(PYTHON) scripts/demo_depth_guided_search.py

demo-functional-family-induction:
	$(PYTHON) scripts/demo_functional_family_induction.py

demo-hard-capacity:
	$(PYTHON) scripts/demo_hard_capacity.py

test-active:
	$(PYTHON) -m unittest tests.test_active -v

active-calibrate:
	$(PYTHON) scripts/active_calibrate.py

ASEEDS ?= 16
APROCS ?= 12
active-structure-multiseed:
	$(PYTHON) scripts/active_multiseed.py $(ASEEDS) $(APROCS)

active-structure-ablation:
	$(PYTHON) scripts/active_multiseed.py $(ASEEDS) $(APROCS) main

active-structure-noise-sweep:
	$(PYTHON) scripts/active_multiseed.py $(ASEEDS) $(APROCS) noise

active-structure-scaling:
	$(PYTHON) scripts/active_multiseed.py $(ASEEDS) $(APROCS) main,qp

active-structure-sealed-eval:
	$(PYTHON) scripts/active_sealed_eval.py

# ---- AWS (SageMaker Processing, eu-north-1, profil wce-runner) — reconstruction robuste
AWS_PROFILE ?= wce-runner
AWSPY = AWS_PROFILE=$(AWS_PROFILE) $(PYTHON) -m wce_aws.orchestrate
SEEDS16 = 0-15
KEYS ?= $(SEEDS16)
AWS_SHARDS ?= 1
AWS_INSTANCE ?= $(if $(filter ec2,$(WCE_BACKEND)),t3.large,ml.c5.4xlarge)
.PHONY: aws-smoke aws-smoke-job aws-package aws-status aws-logs aws-fetch-results aws-clean-safe aws-run-robust-structure aws-multiseed aws-noise-sweep aws-run-calibration aws-run-aux aws-sealed-eval robust-aggregate test-robust demo-robust-reconstruction demo-measurement-selection demo-noise-vs-structure demo-compatible-structures demo-active-family-validation
aws-smoke:
	$(AWSPY) smoke

aws-smoke-job:
	$(AWSPY) smoke --submit

aws-package:
	$(AWSPY) package

aws-status:
	$(AWSPY) status $(if $(RUN),--run $(RUN),)

aws-logs:
	$(AWSPY) logs --run $(RUN)

aws-fetch-results:
	$(AWSPY) fetch $(if $(RUN),--run $(RUN),)

aws-clean-safe:
	$(AWSPY) clean-safe

aws-run-robust-structure aws-multiseed:
	$(AWSPY) submit --kind main --keys $(KEYS) --shards $(AWS_SHARDS) --instance $(AWS_INSTANCE)

aws-noise-sweep:
	$(AWSPY) submit --kind noise --keys $(KEYS) --shards $(AWS_SHARDS) --instance $(AWS_INSTANCE)

aws-run-calibration:
	$(AWSPY) submit --kind calibration --keys $(KEYS) --shards $(AWS_SHARDS) --instance $(AWS_INSTANCE)

aws-run-aux:
	for k in hard repeat family capacity identifiability; do $(AWSPY) submit --kind $$k --keys $(KEYS) --shards $(AWS_SHARDS) --instance $(AWS_INSTANCE) || exit 1; done

aws-sealed-eval:
	$(PYTHON) -c "from core.robust_world import sealed_hash, SEALED_SHA256; assert sealed_hash() == SEALED_SHA256, 'lot scellé modifié'"
	$(AWSPY) submit --kind sealed --keys 0-23 --shards $(AWS_SHARDS) --instance $(AWS_INSTANCE)

robust-aggregate:
	$(PYTHON) -m core.robust_aggregate

test-robust:
	$(PYTHON) -m unittest tests.test_robust -v

demo-robust-reconstruction:
	$(PYTHON) scripts/demo_robust_reconstruction.py

demo-measurement-selection:
	$(PYTHON) scripts/demo_measurement_selection.py

demo-noise-vs-structure:
	$(PYTHON) scripts/demo_noise_vs_structure.py

demo-compatible-structures:
	$(PYTHON) scripts/demo_compatible_structures.py

demo-active-family-validation:
	$(PYTHON) scripts/demo_active_family_validation.py

.PHONY: aws-login aws-whoami aws-check-guardrails aws-ec2-diagnose aws-timing aws-inspect aws-stop aws-progress aws-resume aws-profile aws-bench aws-run-caltrain aws-run-rest
aws-login:
	aws sso login --profile $(AWS_PROFILE) --use-device-code

aws-whoami:
	$(AWSPY) whoami

aws-check-guardrails:
	$(AWSPY) check-guardrails

aws-ec2-diagnose:
	$(AWSPY) ec2-diagnose

aws-timing:
	$(AWSPY) timing

aws-inspect:
	$(AWSPY) inspect $(if $(RUN),--run $(RUN),)

aws-stop:
	$(AWSPY) stop --run $(RUN) $(if $(HARD),--hard,)

aws-progress:
	$(AWSPY) progress $(if $(RUN),--run $(RUN),)

aws-resume:
	$(AWSPY) resume --run $(RUN)

# instrumentation (temps, pas de résultat scientifique) : profilage d'une tâche du banc principal, micro-benchmark par type d'instance
aws-profile:
	$(AWSPY) submit --kind profile --keys $(or $(TASK),1) --shards 1 --instance $(AWS_INSTANCE) --max-runtime 7200

aws-bench:
	$(AWSPY) submit --kind bench --keys 0 --shards 1 --instance $(AWS_INSTANCE) --max-runtime 7200

# tout sauf le banc principal et le lot scellé, en UN run (les instances restent pleines sans intervention)
REST_SPEC = noise:$(SEEDS16);hard:$(SEEDS16);repeat:$(SEEDS16);family:$(SEEDS16);capacity:$(SEEDS16);identifiability:$(SEEDS16);calibration:$(SEEDS16);caltrain:0-5
aws-run-rest:
	$(AWSPY) submit --kind "$(REST_SPEC)" --shards $(AWS_SHARDS) --instance $(AWS_INSTANCE) --max-runtime 28800

aws-run-caltrain:
	$(AWSPY) submit --kind caltrain --keys 0-5 --shards $(AWS_SHARDS) --instance $(AWS_INSTANCE)

# ---- jalon Deep Compositional Structure Discovery & Model-Class Boundary
.PHONY: deep-check deep-test
deep-check:
	$(PYTHON) -m compileall -q core/deep_world.py core/deep_structure.py core/deep_fit.py core/deep_search.py core/deep_equivalence.py core/deep_boundary.py core/deep_experiments.py tests/test_deep.py
	$(PYTHON) -c "from core.deep_world import sealed_hash, SEALED_SHA256; assert sealed_hash() == SEALED_SHA256, 'lot scellé modifié'; print('lot scellé intact :', SEALED_SHA256[:16])"

deep-test:
	$(PYTHON) -m unittest tests.test_deep

# Calculs quantitatifs : AWS uniquement (EC2 + SSM explicite, c6i.xlarge, checkpoints S3, reprise, terminaison automatique).
DEEP_BACKEND = WCE_BACKEND=ec2
DEEP_INSTANCE ?= c6i.xlarge
DEEP_SHARDS ?= 2
DEEP_KEYS ?= 0-15
DEEP_PROFILE_KEYS ?= 0
DEEP_DEV_KEYS ?= 0-7
.PHONY: deep-profile deep-run-dev deep-run-val deep-run-multiseed deep-run-sealed deep-progress deep-resume deep-status deep-fetch deep-timing deep-aggregate
deep-profile:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepprofile --keys $(DEEP_PROFILE_KEYS) --shards 1 --instance $(DEEP_INSTANCE) --max-runtime 10800

deep-run-dev:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepdev --keys $(DEEP_DEV_KEYS) --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 21600

# sous-jalon Deep Candidate Generation (dev uniquement ; jamais le lot scellé)
DEEP_GEN_KEYS ?= 0-7
DEEP_COVER_KEYS ?= 0-5
.PHONY: deep-cover-aggregate deep-run-cover deep-run-gen deep-reachability deep-coverage deep-fragments deep-gen-aggregate deep-gen-report deep-profile-generation
deep-run-gen:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepgen --keys $(DEEP_GEN_KEYS) --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 21600

deep-profile-generation:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepgen --keys 100 --shards 1 --instance $(DEEP_INSTANCE) --max-runtime 10800

deep-run-cover:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepcover --keys $(DEEP_COVER_KEYS) --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 21600

deep-reachability:
	$(PYTHON) -m core.deep_gen_report reachability

deep-coverage:
	$(PYTHON) -m core.deep_gen_report coverage

deep-fragments:
	$(PYTHON) -m core.deep_gen_report fragments

deep-cover-aggregate:
	$(PYTHON) -m core.deep_gen_report cover

deep-gen-aggregate deep-aggregate:
	$(PYTHON) -m core.deep_gen_report aggregate

deep-gen-report:
	$(PYTHON) -m core.deep_gen_report report

# sous-jalon Primitive Discovery (dev uniquement)
DEEP_PRIM_KEYS ?= 0-7
.PHONY: deep-train-proposer deep-install-proposer deep-run-prim deep-prim-aggregate deep-prim-report
deep-train-proposer:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepneural --keys 0-1 --shards 1 --instance $(DEEP_INSTANCE) --max-runtime 7200

deep-install-proposer:
	$(PYTHON) -m core.deep_prim_report install

deep-run-prim:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepprim --keys $(DEEP_PRIM_KEYS) --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 21600

deep-prim-aggregate:
	$(PYTHON) -m core.deep_prim_report aggregate

deep-prim-report:
	$(PYTHON) -m core.deep_prim_report report

# sous-jalon Downstream Selection Loss (dev, vrais atomes fournis)
DEEP_DOWN_KEYS ?= 0-15
.PHONY: deep-run-down deep-down-aggregate deep-down-report
deep-run-down:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepdown --keys $(DEEP_DOWN_KEYS) --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 10800

deep-down-aggregate:
	$(PYTHON) -m core.deep_down_report aggregate

deep-down-report:
	$(PYTHON) -m core.deep_down_report report

# sous-jalon Validated Optimizer (dev, vrais atomes fournis)
.PHONY: deep-run-arms deep-run-optimizer deep-vo-aggregate deep-vo-report
deep-run-arms:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepvo --keys 0-15,100-115 --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 10800

deep-run-optimizer:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepopt --keys 0-7 --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 10800

deep-vo-aggregate:
	$(PYTHON) -m core.deep_vo_report aggregate

deep-vo-report:
	$(PYTHON) -m core.deep_vo_report report

# sous-jalon Objective Alignment & Conditional Optimization (dev)
.PHONY: deep-run-objectives deep-run-residual deep-tune-gate deep-run-e2e deep-obj-aggregate deep-obj-report
deep-run-objectives:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepobj --keys 300-307,200-215 --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 10800

deep-run-residual:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepresid --keys 0 --shards 1 --instance $(DEEP_INSTANCE) --max-runtime 3600

deep-tune-gate:
	$(PYTHON) -m core.deep_obj_report tune

deep-run-e2e:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deepe2e --keys 200-215 --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 10800

deep-obj-aggregate:
	$(PYTHON) -m core.deep_obj_report aggregate

deep-obj-report:
	$(PYTHON) -m core.deep_obj_report report

# sous-jalon Robust Primitive Discovery (dev uniquement ; amont : proposition → validation → estimation → ensemble)
DEEP_RP_KEYS ?= 0-15
DEEP_RP_CHAIN_KEYS ?= 400-415
.PHONY: deep-rp-test deep-rp-inspect deep-rp-diagnose deep-rp-tune deep-rp-smoke deep-run-rp deep-run-rp-chain deep-rp-aggregate deep-rp-report
deep-rp-test:
	$(PYTHON) -m unittest tests.test_deep_robust_primitives

deep-rp-smoke:
	$(PYTHON) -m core.deep_rp_report smoke

deep-rp-diagnose:
	$(PYTHON) -m core.deep_rp_report diagnose

deep-rp-chain-smoke:
	$(PYTHON) -m core.deep_rp_report chain-smoke

deep-rp-tune:
	$(PYTHON) -m core.deep_rp_report tune

deep-rp-inspect:
	$(PYTHON) -m core.deep_rp_report inspect

deep-run-rp:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deeprp --keys $(DEEP_RP_KEYS) --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 10800

deep-run-rp-chain:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deeprp --keys $(DEEP_RP_CHAIN_KEYS) --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 21600

deep-rp-aggregate:
	$(PYTHON) -m core.deep_rp_report aggregate

deep-rp-report:
	$(PYTHON) -m core.deep_rp_report report

# sous-jalon Partial-Observation Primitive Parameter Inference (dev uniquement)
DEEP_PO_KEYS ?= 0-15,500-515
.PHONY: deep-po-test deep-po-smoke deep-po-chain-smoke deep-run-po deep-po-aggregate deep-po-report
deep-po-test:
	$(PYTHON) -m unittest tests.test_deep_partial

deep-po-smoke:
	$(PYTHON) -m core.deep_po_report smoke

deep-po-chain-smoke:
	$(PYTHON) -m core.deep_po_report chain-smoke

deep-run-po:
	$(DEEP_BACKEND) $(AWSPY) submit --kind deeppo --keys $(DEEP_PO_KEYS) --shards $(DEEP_SHARDS) --instance $(DEEP_INSTANCE) --max-runtime 21600

deep-po-aggregate:
	$(PYTHON) -m core.deep_po_report aggregate

deep-po-report:
	$(PYTHON) -m core.deep_po_report report

deep-progress:
	$(AWSPY) progress $(if $(RUN),--run $(RUN),)

deep-status:
	$(AWSPY) status $(if $(RUN),--run $(RUN),)

deep-resume:
	$(DEEP_BACKEND) $(AWSPY) resume --run $(RUN)

deep-fetch:
	$(AWSPY) fetch $(if $(RUN),--run $(RUN),)

deep-timing:
	$(AWSPY) timing
