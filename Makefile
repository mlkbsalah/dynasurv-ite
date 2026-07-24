SOURCE := bensalama@ruche.mesocentre.universite-paris-saclay.fr:/workdir/bensalama/DynaSurv/models/
DEST := /Users/malek/TheLAB/DynaSurv/models
CONTAINER_NAME = dynasurv
PROJECT_DIR = ~/repos/dynasurv-ite

.PHONY: sync, delsync

sync:
	rsync -avz --progress $(SOURCE) $(DEST)

delsync:
	rsync -avz --delete --progress $(SOURCE) $(DEST)


build-docker:
	docker build --platform linux/amd64 -t $(CONTAINER_NAME):latest .
	docker save $(CONTAINER_NAME):latest -o $(CONTAINER_NAME).tar

send:
	rsync -avz --progress $(CONTAINER_NAME).tar m-ben-salah@172.18.47.92:${PROJECT_DIR}/

build-and-send: build-docker send

build-apptainer:
	srun --job-name=build-apptainer \
	     --time=30:00 \
	     --mem=24G \
	     --cpus-per-task=12 \
	     --partition=ai \
	     apptainer build $(CONTAINER_NAME).sif docker-archive://$(CONTAINER_NAME).tar

run-interactive:
	srun --job-name=nano-jepa_interactive \
	     --time=12:00:00 \
	     --mem=24G \
	     --cpus-per-task=16 \
	     --partition=ai \
	     --gres=gpu:2 \
	     --pty apptainer shell --nv $(CONTAINER_NAME).sif
