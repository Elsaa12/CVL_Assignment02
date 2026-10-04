"""Membuat tabel LaTeX langsung dari berkas CSV hasil notebook (tanpa mengetik angka manual)."""
import sys, os, json
import pandas as pd

hasil, keluar = sys.argv[1], sys.argv[2]
os.makedirs(keluar, exist_ok=True)

def f(x, d=3):
    return f'{x:.{d}f}'.replace('.', ',')

def tulis(nama, isi):
    with open(os.path.join(keluar, nama), 'w') as fh:
        fh.write(isi)

# ------------------------------------------------------------------ Tabel utama
u = pd.read_csv(os.path.join(hasil, 'hasil_utama.csv'))
baris = []
sebelumnya = None
for _, r in u.iterrows():
    nama = r['Model'].replace('YOLO11', 'YOLO11').replace('zero-shot', r'\textit{zero-shot}').replace('fine-tuned', r'\textit{fine-tuned}')
    kolom_model = nama if r['Model'] != sebelumnya else ''
    if r['Model'] != sebelumnya and sebelumnya is not None:
        baris.append(r'\midrule')
    sebelumnya = r['Model']
    kf = r['Konfigurasi'].replace('tiling+penuh', r'\textit{tiling}+penuh').replace('tiling', r'\textit{tiling}')
    baris.append(' & '.join([kolom_model, kf, f(r['AP50_kecil']), f(r['AP50_sedang']), f(r['AP50_besar']), f(r['AP50_semua']),
                             f(r['AP50:95_kecil']), f(r['AP50:95_semua']), f(r['AR50_kecil']), f(r['ms/citra'], 1)]) + r' \\')
tulis('tabel_utama.tex', r'''\begin{table*}[t]
\caption{AP@0,5 dan AP@0,5:0,95 (agnostik kelas) menurut ukuran objek pada 548 citra VisDrone-DET \textit{val}. K = kecil, S = sedang, B = besar, A = semua ukuran. AR@0,5 adalah \textit{recall} objek kecil tanpa ambang skor. Waktu diukur pada GPU Tesla T4 dan mencakup pembacaan citra.}
\label{tab:utama}
\centering\scriptsize
\begin{tabular}{llcccccccc}
\toprule
 & & \multicolumn{4}{c}{AP@0,5} & \multicolumn{2}{c}{AP@0,5:0,95} & AR@0,5 & \\
\cmidrule(lr){3-6}\cmidrule(lr){7-8}
Model & Konfigurasi & K & S & B & A & K & A & K & ms/citra \\
\midrule
''' + '\n'.join(baris) + r'''
\bottomrule
\end{tabular}
\end{table*}
''')

# ------------------------------------------------------------------ Tabel grid + statistik
g = pd.read_csv(os.path.join(hasil, 'hasil_grid.csv'))
bg = []
for _, r in g.iterrows():
    ket = r['Keterangan'].replace('@', r'\,@\,').replace('~', r'$\sim$')
    bg.append(f"{int(r['S'])} & {int(r['Sel'])} & {ket} & {f(r['Kehilangan (%)'], 1)} \\\\")
tulis('tabel_grid.tex', r'''\begin{table}[t]
\caption{Fraksi objek VisDrone-DET \textit{val} yang kehilangan sel penanggung jawab bila satu sel hanya memuat satu objek.}
\label{tab:grid}
\centering\scriptsize
\begin{tabular}{rrlr}
\toprule
$S$ & Sel & Padanan & Hilang (\%) \\
\midrule
''' + '\n'.join(bg) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ------------------------------------------------------------------ Tabel mAP 10 kelas
m = pd.read_csv(os.path.join(hasil, 'hasil_map10.csv'))
bm = []
for _, r in m.iterrows():
    kf = r['Konfigurasi'].replace('tiling+penuh', r'\textit{tiling}+penuh').replace('tiling', r'\textit{tiling}')
    bm.append(' & '.join([kf, f(r['mAP50_kecil']), f(r['mAP50_sedang']), f(r['mAP50_besar']), f(r['mAP50_semua']), f(r['mAP50:95_semua'])]) + r' \\')
tulis('tabel_map10.tex', r'''\begin{table}[t]
\caption{mAP sadar kelas (10 kelas VisDrone) model YOLO11n \textit{fine-tuned}.}
\label{tab:map10}
\centering\scriptsize
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{4}{c}{mAP@0,5} & mAP@0,5:0,95 \\
\cmidrule(lr){2-5}\cmidrule(lr){6-6}
Konfigurasi & K & S & B & A & A \\
\midrule
''' + '\n'.join(bm) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ------------------------------------------------------------------ Tabel kepadatan
k = pd.read_csv(os.path.join(hasil, 'hasil_kepadatan.csv'))
pilih = [('YOLO11n zero-shot', '640'), ('YOLO11n zero-shot', 'tiling'), ('YOLO11s zero-shot', '640'), ('YOLO11s zero-shot', 'tiling'),
         ('YOLO11n fine-tuned', '640'), ('YOLO11n fine-tuned', 'tiling')]
bk = []
for nm, kf in pilih:
    r = k[(k['Model'] == nm) & (k['Konfigurasi'] == kf)]
    if r.empty:
        continue
    r = r.iloc[0]
    nama = nm.replace('zero-shot', r'\textit{zero-shot}').replace('fine-tuned', r'\textit{fine-tuned}')
    kfl = kf.replace('tiling', r'\textit{tiling}')
    bk.append(' & '.join([nama, kfl, f(r['AP50 kecil | jarang']), f(r['AP50 kecil | sedang']), f(r['AP50 kecil | padat'])]) + r' \\')
tulis('tabel_kepadatan.tex', r'''\begin{table}[t]
\caption{AP@0,5 objek kecil menurut kepadatan citra (tersil jumlah objek per citra).}
\label{tab:kepadatan}
\centering\scriptsize
\begin{tabular}{llccc}
\toprule
Model & Konfigurasi & Jarang & Sedang & Padat \\
\midrule
''' + '\n'.join(bk) + r'''
\bottomrule
\end{tabular}
\end{table}
''')
print('tabel dibuat di', keluar, sorted(os.listdir(keluar)))
