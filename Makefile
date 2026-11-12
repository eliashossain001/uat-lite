# UAT-LITE experiment shortcuts.  `make help` lists targets.
# Override variables on the command line, e.g.  make ood M=20 UAT=embedding
PY     ?= python
M      ?= 10
UAT    ?= attention
DEVICE ?= cuda
IMAGE  ?= uat-lite
EVAL    = --M $(M) --uat $(UAT) --device $(DEVICE)

.PHONY: help install test train train-clinical eval selective shift clinical ood risk-coverage ablations latency figures all docker-build docker-shell

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:          ## install the package and dependencies
	pip install -e .

test:             ## CPU unit tests
	$(PY) -m pytest -q tests

train:            ## fine-tune BERT-base on SST-2, MNLI, SQuAD 2.0
	$(PY) scripts/train/train.py --dataset sst2  --epochs 3 --batch_size 16 --gradient_accumulation_steps 1 --max_length 128 --seed 42
	$(PY) scripts/train/train.py --dataset mnli  --epochs 3 --batch_size 8  --gradient_accumulation_steps 2 --max_length 256 --seed 42
	$(PY) scripts/train/train.py --dataset squad --epochs 3 --batch_size 8  --gradient_accumulation_steps 2 --max_length 384 --seed 42

train-clinical:   ## fine-tune Bio_ClinicalBERT on MedQA / PubMedQA and evaluate
	$(PY) scripts/evaluate/clinical.py $(EVAL)

selective:        ## selective prediction on MNLI and SQuAD 2.0
	$(PY) scripts/evaluate/selective_prediction.py --task mnli  $(EVAL)
	$(PY) scripts/evaluate/selective_prediction.py --task squad $(EVAL)

shift:            ## MNLI matched -> mismatched shift (5 seeds)
	$(PY) scripts/evaluate/mnli_shift.py $(EVAL)

clinical:         ## clinical QA evaluation from existing checkpoints
	$(PY) scripts/evaluate/clinical.py --eval_only $(EVAL)

ood:              ## HANS, ANLI, SNLI, IMDb
	$(PY) scripts/ood/hans.py $(EVAL)
	$(PY) scripts/ood/anli.py --skip_hans $(EVAL)
	$(PY) scripts/ood/snli.py $(EVAL)
	$(PY) scripts/ood/imdb.py $(EVAL)

risk-coverage:    ## per-example dumps + risk-coverage curves
	$(PY) scripts/ood/dump_per_example.py $(EVAL)
	$(PY) scripts/ood/risk_coverage.py --in_dir results/per_example --out_dir results --fig_dir results/figures

eval: selective shift clinical ood   ## all main evaluations

ablations:        ## Q/K/V placement, lambda, penalty form, negation, behavioral buckets
	for t in sst2 mnli squad; do $(PY) scripts/analysis/qkv_ablation.py --task $$t --M $(M) --device $(DEVICE); done
	$(PY) scripts/analysis/lambda_sweep.py --M $(M) --device $(DEVICE)
	$(PY) scripts/analysis/penalty_form.py --task squad --M $(M) --device $(DEVICE)
	$(PY) scripts/analysis/negation.py --M $(M) --device $(DEVICE)
	$(PY) scripts/analysis/behavioral.py --dataset medqa --ckpt_path experiments/medqa/bert-base-uncased/checkpoints/best_model.pt $(EVAL)

latency:          ## GPU latency per method
	$(PY) scripts/efficiency/gpu_latency.py --M $(M)

all: train train-clinical eval ablations latency   ## everything, from scratch

docker-build:     ## build the Docker image
	docker build -t $(IMAGE) .

docker-shell:     ## interactive shell in the container (GPU, mounted checkpoints/results)
	docker run --rm -it --gpus all -v $(PWD)/experiments:/workspace/uat-lite/experiments \
	  -v $(PWD)/results:/workspace/uat-lite/results -v $(HOME)/.cache/huggingface:/workspace/hf_cache $(IMAGE)
