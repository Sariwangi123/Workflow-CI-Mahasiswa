# Sumber dataset

Predict Students' Dropout and Academic Success, UCI ID 697.

- Halaman: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- DOI: https://doi.org/10.24432/C5MC89
- Arsip resmi: https://archive.ics.uci.edu/static/public/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess.zip
- Diunduh: 2 November 2026.
- Lisensi: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Atribusi: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository.
- SHA256 data.csv: `3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae`

File mentah dipertahankan tanpa modifikasi. Dataset berasal dari institusi pendidikan tinggi di Portugal; bukan mahasiswa Indonesia. Tiga label adalah status akhir durasi normal studi, bukan diagnosis risiko individual. Penggunaan model di kampus lain memerlukan validasi lokal.

Rancangan prediksi: akhir semester pertama. Seluruh fitur semester kedua dikeluarkan. Debtor, Tuition fees up to date, dan tiga indikator ekonomi dikeluarkan karena waktu pencatatannya belum terverifikasi untuk cutoff tersebut. Fitur kode nominal diperlakukan sebagai kategori. Baris duplikat ditangani sebelum split; jumlah perubahan dicatat dalam manifest. Transformasi dipelajari hanya pada train.
