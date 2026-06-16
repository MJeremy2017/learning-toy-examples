---
name: neural-net-tuning
description: Diagnose and tune neural network training using the loss/metric curves. Use when tuning a neural network, choosing hyperparameters (learning rate, batch size, depth, width, dropout, weight decay, sequence length), diagnosing overfitting/underfitting/instability, or deciding how to change a model architecture based on training results.
---

# Neural Network Tuning

Read the training curves first, then change ONE knob per run so each result is interpretable. The curves tell you which lever to pull.

## Workflow

```
- [ ] 1. Trust the measurement (full data, fixed seed, right metric, chronological val for time series)
- [ ] 2. Read the curves (gap, trajectory, stopping epoch, stability)
- [ ] 3. Diagnose with the decision tree
- [ ] 4. Change ONE knob, re-run, log (change, val_metric, best_epoch)
- [ ] 5. Repeat in the tuning order below
```

## Step 1: Trust the measurement first

- **Full dataset**, not a smoke-test subset (check sample counts).
- **Fixed seed** so run-to-run differences come from the change, not noise.
- Track the **metric you actually care about** (same as any baseline you compare to).
- Held-out **val/OOF split that matches real evaluation** — for time series, chronological, never shuffled across time.

## Step 2: Diagnose (gap first, then the train level / trajectory)

Judge "good vs. mediocre" against a baseline or known ceiling, **not** against zero — in low-signal domains (e.g. finance) small numbers can still be good.

```
Big gap (train ≫ val)?
├─ Train fits well (near ceiling)  → Overfitting        → ↓ capacity, ↑ regularization, more data
└─ Train mediocre                  → Low signal / shift → better/more-stationary features + more data; keep capacity
Small gap (train ≈ val)?
├─ Both still improving            → Healthy/undertrained → train longer (↑ epochs/patience)
└─ Both plateaued at a poor level  → Underfitting        → ↑ capacity, ↓ regularization, fix LR/inputs
```

Key nuances:
- A gap alone does **not** mean "too big." Underfitting is the *plateau* (train stops improving), not just "both look low."
- Both curves still **decreasing in sync** = healthy/undertrained → just train longer. Not underfitting.
- **Guiding principle:** set capacity from the train curve; close the gap with regularization/data. Shrinking caps achievable performance — prefer soft regularizers (dropout, weight decay, early stopping, more data, auxiliary targets). Only shrink when train is strong and val lags; only grow when train is weak and val still has headroom.

Instability cases (outside the gap tree):

| What you see | Likely cause | Fix |
|---|---|---|
| loss NaN / bounces / unstable | LR too high or inputs unscaled | ↓ LR, standardize inputs, grad clip (max_norm ~1.0) |
| best epoch 1–2 then stops | LR too high / fast overfit | ↓ LR, add LR scheduler |
| val worse than trivial baseline | scale/LR bug or too-short training | fix normalization + LR first, then re-evaluate |

## Step 3: Tuning order (one knob at a time)

1. **Learning rate** — most important. Sweep by ~3x (3e-4, 1e-3, 3e-3) + a scheduler (ReduceLROnPlateau/cosine).
2. **Capacity** — width (hidden size) first, then depth. Scale up only while val keeps improving.
3. **Context** — sequence length for RNNs/Transformers, receptive field for CNNs.
4. **Regularization** — dropout, weight decay. Add only to close an overfitting gap you actually observe.
5. **Batch size** — a speed/memory knob, not a quality knob. Raise it a lot → raise LR roughly proportionally.

Keep a per-run log: `(change, val_metric, best_epoch, notes)`.

## High-impact items people forget

- **Input normalization** with **train-split stats only** — often the biggest single quality/stability win for NNs.
- **LR schedule** — decay when val stalls; a fixed LR leaves performance on the table.
- **Early stopping + restore best weights** — stop on the val metric, keep the best checkpoint.
- **Gradient clipping** (max_norm ~1.0) — stabilizes RNNs/Transformers.
- **Match the loss to the objective** and any sample weighting; clip/transform targets to a sane range.
- **Seed** Python/NumPy/framework RNGs so comparisons are valid.

## Architecture change guidance

- Add **depth/width** only when a smaller model underfits the **train** set *and* val still has headroom. A gap with mediocre train is not a reason to add capacity — fix inputs/regularization first.
- For sequences, **never use bidirectional/future context in forecasting** — it leaks the target.
- Prefer a **cheaper variant first** (e.g. GRU vs LSTM) once context length is set.
- Change architecture **after** LR, normalization, and capacity are dialed in — not before.

## Anti-patterns

- Changing several knobs at once (you learn nothing).
- Tuning on a dev subset, then being surprised by full-data results.
- Chasing train loss instead of the val metric.
- Shuffling time-series data across the split boundary.
- Adding regularization before confirming overfitting exists.
