import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# KONFIGURASI FOLDER
# =========================================================
# Folder sumber gambar: "gambar 3"
FOLDER_GAMBAR = Path("gambar_3")

# Hasil disimpan terpisah agar grayscale dan histogram
# tidak saling menimpa (overlap/overwrite).
FOLDER_HASIL = Path("hasil tugas_3")
FOLDER_GRAYSCALE = FOLDER_HASIL / "grayscale"
FOLDER_HISTOGRAM = FOLDER_HASIL / "histogram"

FOLDER_GRAYSCALE.mkdir(parents=True, exist_ok=True)
FOLDER_HISTOGRAM.mkdir(parents=True, exist_ok=True)

# Ambil semua file gambar yang didukung dari folder sumber.
ekstensi_gambar = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
daftar_gambar = sorted(
    p for p in FOLDER_GAMBAR.iterdir()
    if p.is_file() and p.suffix.lower() in ekstensi_gambar
) if FOLDER_GAMBAR.exists() else []

if not FOLDER_GAMBAR.exists():
    raise FileNotFoundError(
        f"Folder sumber '{FOLDER_GAMBAR}' tidak ditemukan. "
        "Pastikan folder 'gambar 3' berada satu folder dengan file Python ini."
    )

if not daftar_gambar:
    raise FileNotFoundError(
        f"Tidak ada file gambar yang didukung di folder '{FOLDER_GAMBAR}'."
    )

# =========================================================
# PROSES SETIAP GAMBAR SEKALI
# =========================================================
for nomor, path_gambar in enumerate(daftar_gambar, start=1):
    gambar = cv2.imread(str(path_gambar))

    if gambar is None:
        print(f"[Lewati] Gambar tidak dapat dibaca: {path_gambar.name}")
        continue

    # Satu kali konversi grayscale dipakai untuk menyimpan grayscale
    # sekaligus membuat histogram grayscale.
    gambar_rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------
    # 1. Simpan hasil grayscale
    # -----------------------------------------------------
    nama_grayscale = FOLDER_GRAYSCALE / f"{path_gambar.stem}_grayscale.png"
    berhasil = cv2.imwrite(str(nama_grayscale), grayscale)

    if berhasil:
        print(f"[Grayscale] {nama_grayscale}")
    else:
        print(f"[Gagal] Tidak bisa menyimpan {nama_grayscale}")

    # -----------------------------------------------------
    # 2. Buat dan simpan histogram RGB + grayscale
    # -----------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Histogram RGB
    warna = ["Red", "Green", "Blue"]
    for kanal, label in enumerate(warna):
        axes[0].hist(
            gambar_rgb[:, :, kanal].ravel(),
            bins=256,
            range=(0, 256),
            alpha=0.5,
            label=label
        )

    axes[0].set_title(f"Histogram Citra RGB - {path_gambar.name}")
    axes[0].set_xlabel("Intensitas Piksel")
    axes[0].set_ylabel("Jumlah Piksel")
    axes[0].set_xlim(0, 255)
    axes[0].set_xticks(range(0, 256, 10))
    axes[0].legend()

    # Histogram grayscale
    axes[1].hist(
        grayscale.ravel(),
        bins=256,
        range=(0, 256),
        color="gray"
    )
    axes[1].set_title(f"Histogram Grayscale - {path_gambar.name}")
    axes[1].set_xlabel("Intensitas Piksel")
    axes[1].set_ylabel("Jumlah Piksel")
    axes[1].set_xlim(0, 255)
    axes[1].set_xticks(range(0, 256, 10))

    fig.tight_layout()
    nama_histogram = FOLDER_HISTOGRAM / f"{path_gambar.stem}_histogram.png"
    fig.savefig(str(nama_histogram), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Histogram] {nama_histogram}")

print("\\nSemua gambar selesai diproses.")
print(f"Hasil grayscale : {FOLDER_GRAYSCALE.resolve()}")
print(f"Hasil histogram : {FOLDER_HISTOGRAM.resolve()}")
