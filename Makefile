.PHONY: test baseline compile import-properties demo-idealista demo-user-properties

test:
	PYTHONPATH=src python3 -m unittest discover

baseline:
	PYTHONPATH=src python3 scripts/run_baseline.py

demo:
	python3 run_tfm_demo.py

compile:
	python3 -m compileall src tests scripts

import-properties:
	python3 scripts/import_idealista_csv.py

demo-idealista:
	python3 scripts/import_idealista_csv.py --input data/raw/idealista_magnus_template.csv --output data/processed/properties_from_idealista.json
	python3 run_tfm_demo.py --properties data/processed/properties_from_idealista.json --labels "" --top-k 5

demo-user-properties:
	python3 run_tfm_demo.py --properties data/processed/user_properties.json --labels "" --top-k 5
