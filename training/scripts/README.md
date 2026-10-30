# Training scripts

Eksperimen utama saat ini berada di `training/notebooks/`. Script training
tambahan dapat ditempatkan di direktori ini dan mengimpor helper dari:

```python
from alerce_classifier.models import HybridCNNGRU
from alerce_classifier.preprocessing.pipeline import make_padded_sequences
from alerce_classifier.training.loop import train_model
```
