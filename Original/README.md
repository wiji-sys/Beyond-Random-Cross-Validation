# Proyek Analisis Pendidikan Tinggi Indonesia 2025

Proyek ini menggunakan dataset resmi Kemdiktisaintek yang diterbitkan pada Juli 2026
dan merepresentasikan data pendidikan tinggi tahun 2025.

## Notebook tersedia

1. `notebooks/00_Data_Audit_and_Research_Design.ipynb`
   - memvalidasi pembacaan file;
   - menstandarkan nama kolom;
   - memeriksa kunci duplikat;
   - merekonsiliasi agregasi file sumber terhadap dataset gabungan;
   - mendokumentasikan missing values dan unmatched keys;
   - menetapkan target, exposure, prediktor, dan kelompok validasi.

2. `notebooks/01_Descriptive_and_Inequality_Analysis.ipynb`
   - statistik deskriptif tingkat strata dan wilayah;
   - Gini, Theil T, HHI, top-share, dan dekomposisi ketimpangan;
   - kesenjangan gender, konsentrasi STEM, pencilan robust, dan korelasi wilayah.

Notebook 02 akan ditambahkan setelah Notebook 01 dinyatakan lolos.

## Menjalankan proyek

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
jupyter lab
```

Buka notebook dari folder `notebooks/` dan jalankan seluruh sel secara berurutan.

## Catatan metodologis

`graduate_output_intensity` bukan graduation/completion rate karena jumlah mahasiswa
terdaftar dan jumlah lulusan bukan berasal dari cohort longitudinal yang sama.


## Perbaikan Notebook 01 v2

Notebook 01 v2 tidak lagi membutuhkan import `src/descriptive_metrics.py`.
Lokasi dataset juga dicari otomatis dari working directory dan folder induknya.
Untuk hasil paling stabil, jalankan `jupyter lab` dari folder root proyek.

## Notebook 02

`notebooks/02_Negative_Binomial_Production_Function.ipynb`

Notebook ini:

- membuktikan overdispersion Poisson;
- mengestimasi M0–M5 Negative Binomial NB2;
- menggunakan offset `log(jumlah_terdaftar)`;
- melaporkan IRR dengan standard error cluster-provinsi;
- menjalankan VIF, kalibrasi, residual, dan sensitivity analysis;
- menyimpan model utama dan tabel/figur siap artikel.

Jalankan setelah Notebook 00 dan 01 berstatus PASS.

## Notebook 04

`notebooks/04_STEM_Regional_Analysis.ipynb`

Notebook ini menganalisis komposisi dan konsentrasi regional lulusan STEM, membangun model grouped-binomial untuk proporsi STEM, serta model Negative Binomial untuk graduate-to-entry flow dengan interaksi STEM. Rasio aliran bukan completion rate.

## Notebook 05

`notebooks/05_Robustness_and_Sensitivity.ipynb`

Notebook ini menguji ketahanan hasil Notebook 02–04 melalui specification grid,
ambang denominator, winsorization, covariance alternatif, efek tetap provinsi,
leave-one-province-out, cluster bootstrap, stabilitas ranking prediktif, dan
sensitivitas model STEM.

## Notebook 06

`notebooks/06_Article_Ready_Tables_and_Figures.ipynb`

Notebook ini mengonsolidasikan hasil Notebook 01–05 menjadi tabel utama,
figur utama, materi suplemen, captions, dan Results Ledger yang siap dipakai
untuk penyusunan manuskrip. Notebook tidak menjalankan eksperimen baru.
