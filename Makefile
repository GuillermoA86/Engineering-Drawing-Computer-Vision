install:
	python -m pip install -r requirements.txt

data:
	python scripts/download_dataset.py
	python scripts/inspect_dataset.py
	python scripts/build_manifest.py

train:
	python src/train.py --config configs/resnet18.yaml

evaluate:
	python src/evaluate.py --checkpoint models/best_model.pt

app:
	streamlit run src/app.py

test:
	pytest -q
