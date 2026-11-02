# CI Prediksi Status Studi Mahasiswa

Jalankan `python run_project.py` setelah memasang dependencies MLProject. Launcher memanggil MLflow Project, mengambil ID run yang sebenarnya, dan mengekspor model portable ke `release_artifacts/model`.

GitHub Actions menyimpan artefak run dan membuat GitHub Release berisi `student-model.tar.gz`. Release dipilih agar reviewer dapat mengakses model selain artifact Actions yang memiliki masa retensi. Push image Docker bersyarat `BUILD_DOCKER_IMAGE=true`, dengan secrets `DOCKERHUB_USERNAME` dan `DOCKERHUB_TOKEN`.

Build memakai `mlflow models build-docker --env-manager local`. MLflow memilih image Python sesuai metadata model dan memasang requirements di image tersebut. Opsi ini menghindari builder virtualenv lama yang kembali memakai Ubuntu/Python 3.9. Namespace image saat ini `rizkasuhana/student-study-status`, dengan tag commit sumber.

Serving sebaiknya memakai model dari release hasil CI; untuk validasi lokal gunakan ekspor run Project lokal dengan manifest asal run yang dicatat. Artefak lokal bukan bukti bahwa workflow GitHub sudah berhasil.
