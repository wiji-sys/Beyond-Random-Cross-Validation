# Ringkasan Hasil Notebook 03

- Evaluasi utama menggunakan province-disjoint GroupKFold pada 34 provinsi.
- Model terbaik berdasarkan province-level RMSLE adalah **TweedieRate**.
- GroupCV RMSLE: **0.2255** (bootstrap 95% CI 0.1706–0.2682).
- Random CV RMSLE: **0.2226**.
- Generalization gap: **0.0030**; GroupCV/RandomCV ratio **1.013**.
- GroupCV province MAE: **11,260.18** lulusan.
- GroupCV log-R²: **0.9643**.
- Predicted/observed total ratio: **1.0347**.
- Random CV memiliki overlap provinsi pada seluruh fold, sedangkan GroupCV tidak memiliki overlap.
- Hasil random CV hanya dipakai untuk mendemonstrasikan optimisme validasi, bukan memilih model.
