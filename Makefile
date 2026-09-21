HPC = m-ben-salah@172.18.47.92
CONTAINER_NAME = dynasurv

HPC_PROJECT_PATH = ~/repos/dynasurv-ite
LOCAL_PROJECT_PATH = /Users/malek/TheLAB/DynaSurv



SOURCE := ${HPC_PROJECT_PATH}/models/
DEST := ${LOCAL_PROJECT_PATH}/models

.PHONY: sync, delsync

sync:
	rsync -avz --progress $(HPC):$(SOURCE) $(DEST)

delsync:
	rsync -avz --delete --progress $(HPC):$(SOURCE) $(DEST)


build-docker:
	docker build --platform linux/amd64 -t $(CONTAINER_NAME):latest .
	docker save $(CONTAINER_NAME):latest -o $(CONTAINER_NAME).tar

send:
	rsync -avz --progress $(CONTAINER_NAME).tar $(HPC):${HPC_PROJECT_PATH}/

build-and-send: build-docker send

build-apptainer:
	sbatch --job-name=build-apptainer \
	       --time=30:00 \
	       --mem=24G \
	       --cpus-per-task=12 \
	       --partition=ai \
	       --wrap="apptainer build $(CONTAINER_NAME).sif docker-archive://$(CONTAINER_NAME).tar"

run-interactive:
	srun --job-name=dynasurv_interactive \
	     --time=12:00:00 \
	     --mem=24G \
	     --cpus-per-task=16 \
	     --partition=ai \
	     --gres=gpu:2 \
	     --pty apptainer shell --nv $(CONTAINER_NAME).sif
