.PHONY: install demo models data split train-baseline train eval charts islands examples ui capture test lint

EVAL = docs/eval
RUN = python -m brats_report.evaluate run --data data/brats_subset --split data/brats_split.json
POSTS = largest,min1cm3,all
POST = largest

install:
	pip install -e ".[dev,ui]"

demo:                       ## synthetic phantom -> report, no data or model needed
	brats-report --demo --out reports/demo

models:                     ## trained models from the GitHub release (SHA-256 checked)
	python scripts/download_models.py          # add --all for the baseline U-Net (make eval)

data:                       ## 120 BraTS patients: ~1.3 GB of the 7.6 GB archive (range requests)
	python scripts/download_brats.py --n 120 --prefix-mb 1300 --out data/brats_subset

split:                      ## fixed 80 / 10 / 30 split; the baseline's 30 cases stay out of test
	python -m brats_report.evaluate split --data data/brats_subset --test 30 --val 10 \
		--not-in-test data/baseline_cases.txt --out data/brats_split.json

train-baseline:             ## step 1: 30 patients (24 train / 6 val), 12 epochs
	python -m brats_report.train --data data/brats_subset --cases data/baseline_cases.txt \
		--epochs 12 --out models/brats_unet_baseline.pt

train:                      ## step 3: the 80 training patients of the split, same recipe
	python -m brats_report.train --data data/brats_subset --split data/brats_split.json \
		--epochs 8 --out models/brats_unet.pt

eval:                       ## every step on the 30 test patients, all 3 clean-ups per run
	$(RUN) --model models/brats_unet_baseline.pt --post $(POSTS) --name "24 patients" \
		--json $(EVAL)/01_baseline.json
	$(RUN) --model models/brats_unet_baseline.pt --tta --post $(POSTS) \
		--name "24 patients + TTA" --json $(EVAL)/02_tta.json
	$(RUN) --model models/brats_unet.pt --post $(POSTS) --name "80 patients" \
		--json $(EVAL)/03_more_data.json
	$(RUN) --model models/brats_unet.pt --tta --post $(POSTS) --name "80 patients + TTA" \
		--json $(EVAL)/04_more_data_tta.json

charts:                     ## README charts + results table (the app's clean-up: POST)
	python -m brats_report.charts --eval $(EVAL)/01_baseline_$(POST).json \
		$(EVAL)/02_tta_$(POST).json $(EVAL)/03_more_data_$(POST).json \
		$(EVAL)/04_more_data_tta_$(POST).json --final 3 \
		--classifier $(EVAL)/classifier.json --out docs/images

islands:                    ## how separate pieces inflated diameters, on the expert masks
	python scripts/islands_effect.py --json $(EVAL)/islands.json
	python scripts/islands_figure.py --case BRATS_131 --out docs/images/islands_before_after.png

examples:                   ## held-out cases for the UI's example list
	python scripts/make_examples.py

ui:
	python app.py

capture:                    ## README GIF + screenshots (app must be running; `.[demo]` extra)
	python scripts/capture_demo.py

test:
	pytest -q

lint:
	ruff check src scripts tests app.py
