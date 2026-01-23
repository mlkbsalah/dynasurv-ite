SOURCE := bensalama@ruche.mesocentre.universite-paris-saclay.fr:/workdir/bensalama/DynaSurv/models/
DEST := /Users/malek/TheLAB/DynaSurv/models

.PHONY: sync

sync:
	rsync -avz --progress $(SOURCE) $(DEST)