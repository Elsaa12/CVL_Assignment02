# Menguji Keterbatasan YOLO pada Objek Kecil yang Berkelompok di VisDrone-DET

Penugasan individu mata kuliah **Computer Vision**, Magister Kecerdasan Artifisial, Universitas Gadjah Mada.
Opsi C: YOLO dan VisDrone-DET.

| | |
|---|---|
| Penulis | Elsa Aiziyah |
| Paper yang diuji | Redmon dkk. (2016), *You Only Look Once: Unified, Real-Time Object Detection*, Bagian 2.4 (keterbatasan YOLO) |
| Dataset | VisDrone-DET 2019 (10 kategori objek, citra dari drone) |
| Model | YOLO11n *zero-shot*, YOLO11s *zero-shot*, YOLO11n *fine-tuned* |
| Tautan Colab | *(https://drive.google.com/file/d/1wVPXD3opS62941b6djKesyzahrbxxBJk/view?usp=sharing)* |

---

## 1. Ringkasan

Redmon dkk. menyatakan bahwa YOLO sulit mendeteksi objek kecil yang muncul berkelompok. Penyebabnya, setiap sel grid hanya memprediksi dua kotak dan satu kelas, fitur yang dipakai relatif kasar, dan *loss* menyamakan galat pada kotak kecil dan besar. Proyek ini menguji klaim tersebut pada detektor YOLO modern dengan tiga pertanyaan penelitian.

| Kode | Pertanyaan | Cara pengujian |
|---|---|---|
| RQ1 | Seberapa sering batas satu objek per sel grid terlampaui? | Menyandikan target YOLOv1 pada grid S = 7 sampai 160 dan menghitung objek yang tertimpa. |
| RQ2 | Apakah YOLO11 masih lemah pada objek kecil dan citra padat? | AP@0,5 dan AP@0,5:0,95 menurut ukuran objek (kecil, sedang, besar) dan menurut kepadatan citra. |
| RQ3 | Apakah menaikkan resolusi atau *tiling* mengurangi kelemahan itu? | Resolusi masukan 640, 960, 1280, dan *tiling* (irisan 640, tumpang tindih 20%, NMS 0,5, ditambah varian citra penuh). |

## 2. Hasil utama

Seluruh angka berasal dari notebook yang dijalankan di Google Colab (GPU T4) pada 548 citra VisDrone-DET *val*. Evaluasi memakai metrik COCO dengan mode agnostik kelas.

| Model | Konfigurasi | AP@0,5 kecil | AP@0,5 besar | ms/citra |
|---|---|---|---|---|
| YOLO11n *zero-shot* | 640 | 0,171 | 0,871 | 17,8 |
| YOLO11n *zero-shot* | 1280 | 0,362 | 0,894 | 22,5 |
| YOLO11n *zero-shot* | *tiling* | 0,341 | 0,857 | 85,4 |
| YOLO11s *zero-shot* | 640 | 0,256 | 0,917 | 18,2 |
| YOLO11s *zero-shot* | 1280 | 0,448 | 0,938 | 37,8 |
| YOLO11s *zero-shot* | *tiling* | 0,410 | 0,886 | 98,3 |
| YOLO11n *fine-tuned* | 640 | 0,290 | 0,947 | 18,6 |
| YOLO11n *fine-tuned* | 1280 | 0,471 | 0,901 | 22,6 |
| YOLO11n *fine-tuned* | *tiling* | 0,462 | 0,873 | 86,4 |

Temuan pokok:

- Pada resolusi 640, AP@0,5 objek kecil jauh di bawah objek besar pada ketiga model. Klaim paper tetap berlaku pada YOLO11.
- Rasio AP@0,5:0,95 terhadap AP@0,5 untuk objek kecil adalah 0,37, sedangkan untuk objek besar 0,82. Lokalisasi objek kecil jauh lebih rapuh.
- Menaikkan resolusi ke 1280 memberi kenaikan terbesar pada objek kecil. Pada model *fine-tuned*, AP@0,5 objek kecil naik dari 0,290 menjadi 0,471.
- *Tiling* tidak melampaui resolusi 1280 pada ketiga model, dan 2,6 sampai 3,8 kali lebih lambat.
- Penurunan AP@0,5 objek kecil dari citra jarang ke citra padat pada model *fine-tuned* adalah 20,1% (640), 6,9% (1280), dan 1,1% (*tiling*).
- Kasus gagal (citra `0000155_00401_d_0000001`): pada 640 terdeteksi 36 dari 122 objek, dan dengan *tiling* 43 dari 122. Sebanyak 79 objek (64,8%) tetap terlewat pada 640.

Tabel lengkap tersedia di folder `hasil/` dan di laporan.

## 3. Isi paket

```
.
├── README.md
├── Tugas_CV_YOLO_VisDrone.ipynb          Notebook Colab (sudah dijalankan, keluaran tersimpan)
├── Laporan_Tugas_CV_YOLO_VisDrone.docx   Laporan IEEE Conference dua kolom (templat Word)
├── Laporan_Tugas_CV_YOLO_VisDrone.pdf    Laporan versi LaTeX (IEEEtran)
├── hasil/                                Keluaran notebook: tabel (CSV) dan gambar (PNG)
│   ├── hasil_utama.csv                   AP menurut ukuran untuk tiap model dan konfigurasi
│   ├── hasil_kepadatan.csv               AP menurut kepadatan citra
│   ├── hasil_map10.csv                   mAP 10 kelas, model fine-tuned
│   ├── hasil_grid.csv                    Analisis grid YOLOv1 (RQ1)
│   ├── recall_ukuran.csv, recall_kelas.csv
│   ├── sensitivitas_iou.csv, resolusi_efektif.csv
│   ├── statistik_data.csv                Statistik dataset latih dan uji
│   └── g1 … g6 *.png                     Gambar yang dipakai di laporan
└── laporan/                              Sumber LaTeX laporan
    ├── main.tex, bagian_awal.tex, refs.bib
    ├── buat_tabel.py                     Membuat tabel .tex langsung dari CSV
    ├── tabel/                            Tabel hasil pembuatan buat_tabel.py
    ├── IEEEtran.cls, IEEEtran.bst
    └── g1 … g6 *.png
```

Laporan Word dan PDF memuat isi yang sama. Selisih jumlah halaman (4 dan 5) hanya berasal dari perbedaan tata letak templat. Berkas Word mengikuti templat IEEE yang diberikan dan sebaiknya menjadi versi yang dikumpulkan.

## 4. Struktur notebook

| Bagian | Isi |
|---|---|
| 0 | Alur eksperimen dan kaitannya dengan paper |
| 1 | Setup, seed (42), dan profil komputasi |
| 2 | Unduh data dan bobot otomatis dari rilis GitHub Ultralytics |
| 3 | Anotasi dan statistik dataset |
| 4 | RQ1: analisis struktural grid YOLOv1 |
| 5 | Sensitivitas IoU terhadap ukuran kotak dan resolusi efektif |
| 6 | Evaluator AP gaya COCO beserta pengujian mandiri |
| 7 | Fungsi inferensi: resolusi masukan dan *tiling* |
| 8 | *Fine-tuning* YOLO11n pada subset VisDrone |
| 9 | Evaluasi seluruh konfigurasi (RQ2 dan RQ3) |
| 10 | Rincian menurut kepadatan citra |
| 11 | mAP per kelas (10 kelas) |
| 12 | Analisis kasus gagal |
| 13 | Ringkasan, penyimpanan hasil, dan catatan keterbatasan |

## 5. Cara menjalankan

### Notebook

1. Unggah `Tugas_CV_YOLO_VisDrone.ipynb` ke Google Colab, atau buka dari tautan di bagian atas.
2. Pilih *runtime* **GPU (T4)**. Notebook juga dapat berjalan di CPU, tetapi jauh lebih lambat.
3. Jalankan seluruh sel berurutan (*Runtime* → *Run all*). Dataset dan bobot YOLO11 diunduh otomatis, sehingga tidak ada berkas yang perlu disiapkan.
4. Seluruh tabel dan gambar disimpan ke folder `hasil/` di *runtime*.

Konfigurasi yang dilaporkan (profil `ringan`):

| Parameter | Nilai |
|---|---|
| Citra latih | 1000 (subset acak VisDrone-DET *train*, seed 42) |
| Citra *hold-out* | 50 (pemilihan *checkpoint*) |
| Epoch | 12 |
| Batch | 8 |
| Citra uji | seluruh 548 citra VisDrone-DET *val* |

Profil `uji` hanya untuk memeriksa bahwa alur berjalan dan tidak dipakai untuk melaporkan hasil.

Lingkungan yang dipakai: Python 3.13.15, PyTorch 2.11.0+cu130, Ultralytics 8.4.172. Pelatihan memerlukan sekitar 104 menit dan seluruh inferensi sekitar 6 menit pada T4.

### Laporan LaTeX

```bash
cd laporan
python buat_tabel.py ../hasil tabel      # opsional: membuat ulang tabel dari CSV
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Hasilnya `main.pdf`. Diperlukan distribusi TeX dengan paket `babel` (bahasa Indonesia) dan kelas `IEEEtran`. Berkas `IEEEtran.cls` dan `IEEEtran.bst` sudah disertakan.

## 6. Keterbatasan

- **Pelatihan singkat.** Model *fine-tuned* hanya dilatih pada 1000 citra selama 12 epoch dan belum konvergen (mAP@0,5 pada *hold-out* epoch terakhir 0,176). Angka mutlaknya bukan batas atas YOLO11, dan perbandingan *fine-tuned* dengan *zero-shot* tidak setara.
- **Satu kali jalan.** Setiap konfigurasi dijalankan sekali dengan seed 42, tanpa interval kepercayaan.
- **Ambang evaluasi.** Perbandingan antar-konfigurasi memakai ambang yang sama (`conf = 0,001`, IoU NMS 0,7, maksimum 500 deteksi per citra). Hasil dapat bergeser bila ambang diubah.
- **Waktu inferensi** diukur pada satu GPU T4 dan bergantung pada perangkat keras dan beban *runtime* Colab.

Catatan keterbatasan lengkap beserta saran pengembangan ada di notebook (bagian 13) dan di laporan (bagian Kesimpulan dan Keterbatasan).

## 7. Rujukan

- J. Redmon, S. Divvala, R. Girshick, dan A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," *CVPR*, 2016.
- D. Du dkk., "VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results," *ICCVW*, 2019.
- G. Jocher dan J. Qiu, *Ultralytics YOLO11*, 2024.

Daftar pustaka lengkap ada di `laporan/refs.bib` dan pada bagian Pustaka laporan.
