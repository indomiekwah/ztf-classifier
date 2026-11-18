# Training scripts

Eksperimen utama saat ini berada di `training/notebooks/`. Script training
tambahan dapat ditempatkan di direktori ini dan mengimpor helper dari:

```python
from alerce_classifier.models import HybridCNNGRU
from alerce_classifier.preprocessing.pipeline import make_padded_sequences
from alerce_classifier.training.loop import train_model
```

## Evaluasi detection dan benchmark

Setelah training menghasilkan checkpoint, struktur yang digunakan adalah:

```text
artifacts/
├── derived_hybrid_cnn_gru_seed42_best.pt
└── models/
    ├── cnn/derived_cnn_seed42_best.pt
    └── gru/derived_gru_seed42_best.pt
```

Jalankan benchmark dengan:

```powershell
python training\scripts\evaluate_detection_and_benchmark.py `
  --artifact-dir artifacts `
  --split data\alerce_ztf_lc_sanchezsaez_full\data\full\preprocessing\feature_experiment\test_derived_padded.npz `
  --output artifacts\evaluation\benchmark\derived_detection_benchmark_results.json
```

Script mengevaluasi prefix temporal berbasis 6, 10, 15, 20, 30, dan seluruh
detection valid. Hasil mencakup jumlah objek valid, coverage, lima metrik
klasifikasi, parameter trainable, waktu inferensi, peak memory device, ukuran
checkpoint, dan waktu training jika metadata tersebut tersedia pada checkpoint.
