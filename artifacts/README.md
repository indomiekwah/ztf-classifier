# Artifact Layout

The repository's final experiments use raw observation features only.

## Streamlit

The Streamlit application uses:

- `hybrid/hybrid_cnn_gru_seed42_best.pt`
- `normalization.json`
- `feature_config.json`

## Models

Each model directory contains:

- the seed 42 checkpoint;
- full-detection metrics, history, plots, and confusion matrix;
- detection-based early-classification metrics and inference benchmarks.

```text
artifacts/
|-- feature_config.json
|-- normalization.json
|-- class_weights.json
|-- raw_experiment_results.json
|-- benchmark/
|   `-- raw_detection_benchmark_results.json
|-- cnn/
|-- gru/
`-- hybrid/
```

`feature_engineering_experiment.ipynb` and older derived artifacts are not used
as final thesis evidence.
