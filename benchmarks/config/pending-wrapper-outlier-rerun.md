# Pending wrapper-overhead reruns

The previous cleanup removed wrapper-overhead outliers from `raw.csv`. The
StackProcessing rows were rerun successfully. These Python/scikit-image/SciPy
rows still need to be rerun before regenerating final figures:

```sh
PATH="$PWD/.venv-benchmarks/bin:$PATH" bash benchmarks/run_all.sh \
  --skip-builds \
  --repeat 3 \
  --repeat-start 1 \
  --repeat-end 1 \
  --backends python-skimage-scipy \
  --cases benchmarks/config/exact-wrapper-outlier-rerun-cases.csv \
  --operations convolve \
  --pixel-types uint8 \
  --shapes 1024x1024x1024 \
  --parameters kernelSize=7

PATH="$PWD/.venv-benchmarks/bin:$PATH" bash benchmarks/run_all.sh \
  --skip-builds \
  --repeat 3 \
  --repeat-start 2 \
  --repeat-end 2 \
  --backends python-skimage-scipy \
  --cases benchmarks/config/exact-wrapper-outlier-rerun-cases.csv \
  --operations convolve \
  --pixel-types uint16 \
  --shapes 1024x1024x1024 \
  --parameters kernelSize=5
```

After both complete:

```sh
python3 benchmarks/tools/summarize_results.py \
  --input benchmarks/results/raw.csv \
  --output benchmarks/results/summary.csv

MPLCONFIGDIR=/private/tmp/matplotlib-cache \
  .venv-benchmarks/bin/python benchmarks/tools/plot_results.py \
  --input benchmarks/results/summary.csv \
  --output-dir benchmarks/results/figures

cp benchmarks/results/figures/*.pdf notes/LMIP_Optimiser_and_Studio/figures/
```
