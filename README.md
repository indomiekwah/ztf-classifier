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
│   ├── assets/              # gambar landing page dan credit-nya
│   ├── app.py               # halaman Classify object
│   └── pages/               # Result of the research dan About
├── data/                    # dataset lokal
├── artifacts/               # konfigurasi, checkpoint, dan hasil evaluasi
│   ├── models/              # checkpoint CNN dan GRU
│   └── evaluation/          # training, early classification, benchmark
├── pyproject.toml
└── README.md
```

`data/` berisi dataset lokal dan tidak dimaksudkan untuk di-commit. Checkpoint
Hybrid CNN-GRU yang diperlukan aplikasi berada langsung di `artifacts/`;
checkpoint CNN dan GRU untuk benchmark berada di `artifacts/models/`.

## Instalasi

Repository ini menggunakan **Conda untuk membuat environment** dan **pip untuk
memasang package Python**. Pip dijalankan setelah environment Conda aktif.

Untuk instalasi lokal lengkap:

```powershell
conda create -n alerce-dev python=3.10 -y
conda activate alerce-dev
python -m pip install -e ".[dev,training,app]"
```

Perintah `python -m pip install -e ".[dev,training,app]"` memasang repository
secara editable beserta dependency development, training, dan Streamlit.

Untuk environment deployment yang hanya menjalankan aplikasi:

```powershell
conda create -n alerce-app python=3.10 -y
conda activate alerce-app
python -m pip install -e ".[app]"
```

Instalasi editable membuat package `alerce_classifier` dapat digunakan oleh
notebook, script training, dan aplikasi tanpa mengandalkan current working
directory.

## Menjalankan aplikasi Streamlit

```powershell
streamlit run app\app.py
```

Landing page menyediakan randomisasi OID dari
`labeled_set_lc_classifier_SanchezSaez_2020.csv`. Daftar OID dibaca sekali dan
disimpan di cache, sehingga tombol Randomize tidak perlu melakukan query TAP
dan tetap cepat. Query ke ALeRCE baru dilakukan setelah pengguna menekan
Classify.

Jika ingin menambahkan gambar landing page, simpan gambar di:

```text
app/assets/landing_page.png
```

Tambahkan sumber dan credit gambar tersebut di halaman About.

Aplikasi menggunakan checkpoint Hybrid CNN-GRU derived seed 42 dan pipeline
inference yang sama dengan pipeline training. Setelah environment `alerce-app`
aktif, jalankan:

```powershell
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

Data diperoleh menggunakan **Fornax Science Console**. Pengguna Fornax tidak
perlu memasang package repository ini atau membuat environment baru. Gunakan
environment Fornax yang telah tersedia, yaitu `py-light_curve_collector`.
Setelah repository di-download atau
di-clone, salin file berikut ke satu folder kerja di Fornax:

- `src/alerce_classifier/training/fornax_data_acquisition.py`;
- `labeled_set_lc_classifier_SanchezSaez_2020.csv`.
- `environment.yml` jika ingin membuat environment sendiri.

Aktifkan environment `py-light_curve_collector`, buka Terminal Fornax pada
folder yang berisi file tersebut, lalu jalankan script sebagai file Python
biasa. Simpan output penting pada home directory persisten, bukan pada lokasi
scratch yang dapat kedaluwarsa.

Contoh menjalankan akuisisi dari folder kerja Fornax:

```bash
python fornax_data_acquisition.py \
  --per-subclass 2 \
  --output data/smoke
```

Jika repository dipasang secara editable di environment lokal, perintah module
berikut juga dapat digunakan:

```powershell
python -m alerce_classifier.training.fornax_data_acquisition `
  --per-subclass 2 `
  --output data\smoke
```

`environment.yml` dan `fornax_merge.py` yang mungkin digunakan pada workflow
Fornax sebelumnya bukan bagian dari repository ini. Jika workflow tersebut
memerlukannya, sediakan file tersebut secara terpisah.

## Preprocessing dan training

Preprocessing dan training dilakukan melalui package `src/alerce_classifier`, lalu dapat direview melalui notebook di `training/notebooks/`. Dengan pendekatan ini, perubahan preprocessing dan training dapat dievaluasi dan digunakan kembali secara langsung oleh aplikasi.

Model inference yang digunakan aplikasi adalah Hybrid CNN-GRU dengan derived
sequential features.

## Training

Eksperimen utama menggunakan pembagian berbasis OID sebesar **70% training,
15% validation, dan 15% test** dengan stratifikasi `classMacro`. Seluruh
eksperimen menggunakan **random seed 42** untuk menjaga konsistensi dan
reproduktibilitas. Penggunaan satu seed menjadi salah satu keterbatasan
penelitian.

Pipeline menggunakan fitur mentah dan fitur turunan deterministik. Urutan 11
fitur yang digunakan model adalah:

1. `time_rel`
2. `delta_time`
3. `magnitude`
4. `magnitude_error`
5. `flux_log`
6. `flux_error`
7. `delta_magnitude`
8. `local_slope`
9. `log_snr`
10. `filter_g`
11. `filter_r`

Statistik standardisasi setiap fitur dihitung hanya dari detection milik OID
training, kemudian digunakan tanpa refit pada validation dan test.

Metrik yang dilaporkan untuk CNN, GRU, dan Hybrid CNN-GRU adalah
Macro-precision, Macro-recall, Macro-F1, balanced accuracy, dan accuracy.
Model utama dipilih berdasarkan Macro-F1 validation; test set hanya digunakan
satu kali untuk pelaporan hasil akhir.

Early classification utama menggunakan **evaluasi prefix temporal berbasis
jumlah detection** pada 6, 10, 15, 20, 30, dan seluruh detection valid.
Objek yang tidak memiliki detection minimum dikeluarkan dari threshold tersebut
dan jumlah objek valid serta coverage dilaporkan. Evaluasi berbasis persentase
25%, 50%, 75%, dan 100% hanya merupakan analisis tambahan.

```powershell
jupyter notebook training\notebooks\benchmark_experiment.ipynb
```

Notebook tersebut menjalankan setiap tahap benchmark secara terpisah dan
menampilkan tabel metrik, efisiensi, serta evaluasi prefix sebelum menyimpan
artifact JSON.


Benchmark memakai device dan batch size yang sama untuk seluruh model serta
mencatat jumlah parameter, waktu inferensi, peak memory device, ukuran
checkpoint, dan waktu training yang disimpan oleh `train_model`. Jika BRF
tidak dilatih ulang pada dataset ini, BRF diperlakukan sebagai baseline
eksternal dan hanya dibandingkan pada metrik yang tersedia di publikasinya.

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
