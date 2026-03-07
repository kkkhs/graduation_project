# DRENet Local Compatibility Patch Guide

This guide explains how to apply the local compatibility patch used on the training machine.

## Patch File
- `patches/drenet_local_compat_20260307.patch`

## Why This Patch Exists
- PyTorch 2.6 changed `torch.load` defaults (`weights_only=True`) and breaks some legacy paths.
- New NumPy removed `np.int`.
- Torch 2.x is stricter about integer clamp bounds in `loss.py`.

## Patched Files
- `experiments/drenet/DRENet/utils/datasets.py`
- `experiments/drenet/DRENet/utils/general.py`
- `experiments/drenet/DRENet/utils/loss.py`

## Apply On Cloud Or New Host
From project root:

```bash
git apply --verbose patches/drenet_local_compat_20260307.patch
```

If the upstream file content differs and `git apply` fails, use 3-way mode:

```bash
git apply --3way --verbose patches/drenet_local_compat_20260307.patch
```

## Quick Verification
Run:

```bash
grep -n "weights_only=False" experiments/drenet/DRENet/utils/datasets.py
grep -n "weights_only=False" experiments/drenet/DRENet/utils/general.py
grep -n "astype(int)" experiments/drenet/DRENet/utils/datasets.py
grep -n "astype(int)" experiments/drenet/DRENet/utils/general.py
grep -n "gi_max = int" experiments/drenet/DRENet/utils/loss.py
```

## Notes
- The DRENet vendor directory is intentionally ignored in project git history.
- Keep patch files versioned so environment-specific fixes remain reproducible.
