.PHONY: help push-kaggle pull-kaggle-lstm kaggle-status kaggle-status-lstm

KERNEL_LGBM_ID := zhangyue199/train-lgbm-baseline
KERNEL_LSTM_ID := zhangyue199/train-lstm-baseline
KAGGLE_LSTM_DIR := kaggle-lstm

help:
	@echo "Available targets:"
	@echo "  make push-kaggle        Push the LGBM notebook to Kaggle"
	@echo "  make pull-kaggle-lstm   Pull the LSTM notebook from Kaggle"
	@echo "  make kaggle-status      Show the LGBM kernel status"
	@echo "  make kaggle-status-lstm Show the LSTM kernel status"

push-kaggle:
	@test -f kernel-metadata.json || (echo "Missing kernel-metadata.json" && exit 1)
	@test -f train_lgbm_baseline.ipynb || (echo "Missing train_lgbm_baseline.ipynb" && exit 1)
	kaggle kernels push -p .

pull-kaggle-lstm:
	@mkdir -p $(KAGGLE_LSTM_DIR)
	kaggle kernels pull $(KERNEL_LSTM_ID) -p $(KAGGLE_LSTM_DIR) -m

kaggle-status:
	kaggle kernels status $(KERNEL_LGBM_ID)

kaggle-status-lstm:
	kaggle kernels status $(KERNEL_LSTM_ID)
