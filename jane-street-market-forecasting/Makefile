.PHONY: help push-kaggle kaggle-status

KERNEL_ID := zhangyue199/train-lgbm-baseline

help:
	@echo "Available targets:"
	@echo "  make push-kaggle    Push the local notebook to Kaggle"
	@echo "  make kaggle-status  Show the Kaggle kernel status"

push-kaggle:
	@test -f kernel-metadata.json || (echo "Missing kernel-metadata.json" && exit 1)
	@test -f train_lgbm_baseline.ipynb || (echo "Missing train_lgbm_baseline.ipynb" && exit 1)
	kaggle kernels push -p .

kaggle-status:
	kaggle kernels status $(KERNEL_ID)
