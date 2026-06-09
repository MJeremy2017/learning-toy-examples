---
name: neural-net-tuning
description: Diagnose and tune neural network training using the loss/metric curves. Use when tuning a neural network, choosing hyperparameters (learning rate, batch size, depth, width, dropout, weight decay, sequence length), diagnosing overfitting/underfitting/instability, or deciding how to change a model architecture based on training results.
---

# Neural Network Tuning

Tune by reading the training signals first, then change ONE knob per run so each result is interpretable. The model is usually not "broken" — the curves tell you which lever to pull.

## Workflow

```
- [ ] 1. Make the measurement trustworthy (full data, fixed seed, right metric)
- [ ] 2. Read the dashboard (train vs val curve, trajectory, stopping epoch, stability)
- [ ] 3. Diagnose: overfit / underfit / undertrained / unstable
- [ ] 4. Change ONE knob, re-run, log (change, val_metric, best_epoch)
- [ ] 5. Repeat in the tuning order below
```

## Step 1: Trust the measurement first

Before tuning anything, confirm:
- Training on the **full dataset**, not a smoke-test/dev subset (check sample counts).
- A **fixed seed** so run-to-run differences come from the change, not noise.
- Tracking the **same metric you ultimately care about** (and the same one as any baseline you compare against). Know the metric's natural scale — some metrics are tiny by nature, so small gains can be real.
- A held-out **validation/OOF split** that matches the real evaluation (for time series, chronological — never shuffle across time).

## Step 2: The dashboard

Read four things from per-epoch logs and the run summary:

1. **Gap** = train_metric − val_metric → overfit vs underfit.
2. **Val trajectory** → still improving, plateaued, or diverging?
3. **Stopping epoch vs max epochs** → early-stopped, or ran out of budget?
4. **Stability** → loss NaN/exploding, or metric bouncing run-to-run?

## Step 3: Symptom -> diagnosis -> change

| What you see | Diagnosis | Change |
|---|---|---|
| train ≫ val; val peaks early then drops | Overfitting | ↑ dropout, ↑ weight decay, ↓ width/depth, more data/augmentation, keep early stopping |
| train and val both low and close | Underfitting | ↑ width, ↑ depth, ↑ input context (e.g. sequence length), train longer, ↓ regularization |
| best epoch == max epochs, val still rising | Undertrained | ↑ max epochs, ↑ patience |
| best epoch 1-2 then stops | LR too high / fast overfit | ↓ LR, add LR scheduler |
| loss NaN / bounces / metric unstable | LR too high or inputs unscaled | ↓ LR, standardize inputs, add grad clipping (max_norm ~1.0) |
| val worse than trivial baseline | Scale/LR bug or too-short training | fix normalization + LR first, then re-evaluate |

## Step 4: Tuning order (one knob at a time)

1. **Learning rate** — the single most important knob. Sweep by factors of ~3 (e.g. 3e-4, 1e-3, 3e-3). Pair with a scheduler (ReduceLROnPlateau or cosine).
2. **Model capacity** — width (hidden size) first, then depth (layers). Scale up only while val keeps improving.
3. **Architecture-specific context** — e.g. sequence length for RNNs/Transformers, kernel size/receptive field for CNNs.
4. **Regularization** — dropout, weight decay. Add only to close an overfitting gap you actually observe.
5. **Batch size** — mostly a speed/memory knob, not a quality knob. If you raise it a lot, raise LR roughly proportionally.

Keep a small log per run: `(change, val_metric, best_epoch, notes)`.

## High-impact items people forget

- **Input normalization**: standardize features using **train-split statistics only**, then apply to val/test. Critical for NNs (unlike trees, which are scale-invariant). Often the biggest single stability/quality win.
- **LR schedule + warmup**: a fixed LR leaves performance on the table; decay when val stalls.
- **Early stopping + restore best weights**: stop on the val metric, keep the best checkpoint.
- **Gradient clipping**: stabilizes RNNs/Transformers (max_norm ~1.0).
- **Target/output handling**: clip or transform targets to a sane range; match the loss to the objective and any sample weighting.
- **Seed control**: seed Python/NumPy/framework RNGs to make comparisons valid.

## Architecture change guidance

- Add **depth/width** only when a smaller model clearly underfits with headroom on the val curve.
- For sequences, **never use bidirectional/future context in forecasting** — it leaks the target.
- Prefer a **cheaper variant first** (e.g. GRU vs LSTM) as a speed experiment once context length is set.
- Escalate complexity (attention/Transformer, residual blocks, normalization layers) only after exhausting cheaper tuning.
- Change architecture **after** LR, normalization, and capacity are dialed in — not before.

## Anti-patterns

- Changing several knobs at once (you learn nothing).
- Tuning on a dev subset, then being surprised by full-data results.
- Chasing train loss instead of the val metric.
- Shuffling time-series data across the split boundary.
- Adding regularization before confirming overfitting exists.
