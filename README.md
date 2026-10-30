# ALeRCE ZTF Classifier

**Author:** Clara Angelin Pijoh

Repository penelitian skripsi:

> **Implementasi Hybrid 1D-CNN dan GRU untuk Klasifikasi Objek Astronomi
> dengan Kurva Cahaya ALeRCE ZTF**

Sistem menerima kurva cahaya ZTF dari sistem broker [ALeRCE](https://alerce.online) dan menghasilkan prediksi tiga kelas makro:
`Transient`, `Stochastic`, dan `Periodic`.

## Pemetaan kelas

Dataset berisi 15 subkelas berdasarkan penelitian [P. Sánchez-Sáez, dkk. 2021](https://iopscience.iop.org/article/10.3847/1538-3881/abd5c1) pada kolom `classALeRCE`. Subkelas tersebut
dipetakan menjadi `classMacro` sebagai berikut:

| Kelas makro | Subkelas |
|---|---|
| `Transient` | `SNIa`, `SNIbc`, `SNII`, `SLSN` |
| `Stochastic` | `AGN`, `QSO`, `Blazar`, `YSO`, `CV/Nova` |
| `Periodic` | `LPV`, `E`, `DSCT`, `RRL`, `CEP`, `Periodic-Other` |

## Struktur repository

Repository ini menggunakan satu package inti agar preprocessing dan arsitektur
model untuk training dan aplikasi selalu konsisten.

```text
ztf-classifier/
├── src/alerce_classifier/   # package inti yang digunakan bersama
│   ├── clients/             # client TAP dan stamp ALeRCE
│   ├── inference/           # predictor untuk aplikasi
│   ├── models/              # arsitektur Hybrid CNN-GRU
│   ├── preprocessing/       # preprocessing training dan inference
│   └── training/            # helper training, evaluasi, dan akuisisi data
├── training/
│   ├── notebooks/           # eksperimen dan evaluasi
│   ├── scripts/             # script training tambahan
│   └── tests/               # test training
├── app/                     # aplikasi Streamlit
├── data/                    # dataset lokal
├── artifacts/               # konfigurasi, hasil eksperimen, dan checkpoint
├── pyproject.toml
└── README.md
```

`data/` berisi dataset lokal dan tidak dimaksudkan untuk di-commit. Checkpoint
besar juga sebaiknya disimpan menggunakan penyimpanan artifact eksternal.

## Instalasi

Dari root repository, jalankan:

```powershell
python -m pip install -e ".[dev,training,app]"
```

Instalasi editable membuat package `alerce_classifier` dapat digunakan oleh
notebook, script training, dan aplikasi tanpa mengandalkan current working
directory.

## Menjalankan aplikasi Streamlit

```powershell
streamlit run app\app.py
```

Aplikasi menggunakan checkpoint Hybrid CNN-GRU derived seed 42 dan pipeline
inference yang sama dengan pipeline training. Untuk menjalankan aplikasi
produksi/deployment, gunakan environment yang hanya memasang extra `app`:

```powershell
python -m pip install -e ".[app]"
streamlit run app\app.py
```

## Menjalankan notebook

Notebook dijalankan dari root repository agar path `data/` dan `artifacts/`
tetap konsisten:

```powershell
jupyter notebook training\notebooks
```

Notebook mengimpor kode dari package inti, contohnya:

```python
from alerce_classifier.models import HybridCNNGRU
from alerce_classifier.preprocessing.pipeline import prepare_detection_frame
from alerce_classifier.training.loop import train_model
```

## Akuisisi data

Data diperoleh menggunakan **Fornax Science Console** dan
`fornax_data_acquisition`. Untuk menjalankan akuisisi melalui Terminal Fornax,
siapkan:

- `fornax_data_acquisition.py`;
- `environment.yml`;
- `labeled_set_lc_classifier_SanchezSaez_2020.csv`.

Gunakan environment Fornax yang telah tersedia:

```text
py-light_curve_collector
```

Simpan output penting pada home directory persisten, bukan pada lokasi scratch
yang dapat kedaluwarsa. Perintah Conda dijalankan dari Terminal Fornax pada
folder yang berisi ketiga file tersebut.

## Preprocessing dan training

Preprocessing dan training dilakukan melalui package `src/alerce_classifier`, lalu dapat direview melalui notebook di `training/notebooks/`. Dengan pendekatan ini, perubahan preprocessing dan training dapat dievaluasi dan digunakan kembali secara langsung oleh aplikasi.

Model inference yang digunakan aplikasi adalah Hybrid CNN-GRU dengan derived
sequential features.

## Acknowledgement

This work made significant use of the Fornax Science Console, which is part of
the NASA Astrophysics cloud-based Fornax Initiative jointly developed by
Goddard Space Flight Center’s Astrophysics Projects Division (GSFC/ApPD) and
the astrophysics archives: the High Energy Astrophysics Science Archive
Research Center (HEASARC), the Infrared Science Archive (IRSA), and the
Mikulski Archive for Space Telescopes (MAST).

## Dokumentasi sumber

- [ALeRCE TAP](https://science.alerce.online/services/accessing-data-tap/)
- [ALeRCE TAP tables](https://tap.alerce.online/tap/tables)
- [Dataset Sánchez-Sáez di Zenodo](https://zenodo.org/records/4279623)
- [Fornax Science Console](https://docs.fornax.sciencecloud.nasa.gov/quick-start/)
- [Fornax storage](https://docs.fornax.sciencecloud.nasa.gov/data-storage/)
- [Fornax compute environments](https://docs.fornax.sciencecloud.nasa.gov/compute-environments/)

## License

MIT License (non-commercial and can be used with permission).
