.PHONY: build train shell clean help

IMAGE_NAME = pytorch_trainer
PROJECT_DIR = /workdir/DynaSurv

build:
	docker build -t $(IMAGE_NAME) .

train:
	docker run \
	  -v $(shell pwd):$(PROJECT_DIR) \
	  -e PYTHONPATH=$(PROJECT_DIR)/src \
	  -w $(PROJECT_DIR)/scripts \
	  $(IMAGE_NAME) \
	  python train_DynaSurvOnline.py

shell:
	docker run -it \
	  -v $(shell pwd):$(PROJECT_DIR) \
	  -e PYTHONPATH=$(PROJECT_DIR)/src \
	  -w $(PROJECT_DIR) \
	  --entrypoint bash \
	  $(IMAGE_NAME)