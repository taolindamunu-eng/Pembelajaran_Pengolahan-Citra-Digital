import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import os
import csv

# Menentukan folder input dan output
folder_gambar = "gambar_4"
folder_hasil = "hasil_tugas_4"
folder_enhancement = os.path.join(folder_hasil, "enhancement")
folder_histogram = os.path.join(folder_hasil, "histogram")
folder_ocr = os.path.join(folder_hasil, "ocr")

os.makedirs(folder_enhancement, exist_ok=True)
os.makedirs(folder_histogram, exist_ok=True)
os.makedirs(folder_ocr, exist_ok=True)

# Menentukan lokasi Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

nilai_brightness = 50
ekstensi_gambar = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Membaca daftar seluruh gambar
if not os.path.isdir(folder_gambar):
    raise FileNotFoundError(f"Folder '{folder_gambar}' tidak ditemukan.")

daftar_gambar = sorted(
    nama for nama in os.listdir(folder_gambar)
    if nama.lower().endswith(ekstensi_gambar)
    and os.path.isfile(os.path.join(folder_gambar, nama))
)

if not daftar_gambar:
    raise FileNotFoundError(f"Tidak ada gambar di folder '{folder_gambar}'.")

semua_hasil_ocr = []

# Memproses setiap gambar
for nomor, nama_file in enumerate(daftar_gambar, start=1):
    print(f"\nMemproses gambar {nomor}/{len(daftar_gambar)}: {nama_file}")

    path_gambar = os.path.join(folder_gambar, nama_file)
    gambar = cv2.imread(path_gambar)

    if gambar is None:
        print("Gambar tidak dapat dibaca, dilewati.")
        continue

    tinggi, lebar = gambar.shape[:2]
    nama_dasar = os.path.splitext(nama_file)[0]

    # Mengubah gambar menjadi grayscale
    gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # 1. Brightness adjustment
    brightness = cv2.convertScaleAbs(
        gray, alpha=1.0, beta=nilai_brightness
    )

    # 2. Contrast stretching
    min_val = np.min(gray)
    max_val = np.max(gray)

    if max_val != min_val:
        contrast = (
            (gray.astype(np.float32) - min_val)
            * 255 / (max_val - min_val)
        )
        contrast = np.clip(contrast, 0, 255).astype(np.uint8)
    else:
        contrast = gray.copy()

    # 3. Histogram equalization
    histogram_equalization = cv2.equalizeHist(gray)

    daftar_citra = [
        (gray, "Sebelum Enhancement (Grayscale)"),
        (brightness, "Brightness Adjustment"),
        (contrast, "Contrast Stretching"),
        (histogram_equalization, "Histogram Equalization")
    ]

    # Menampilkan dan menyimpan hasil enhancement
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Hasil Enhancement\n" + nama_file,
        fontsize=15, fontweight="bold"
    )

    for ax, (citra, judul) in zip(axes.flat, daftar_citra):
        ax.imshow(citra, cmap="gray")
        ax.set_title(judul, fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")

    fig.subplots_adjust(
        top=0.86, bottom=0.05, left=0.03, right=0.97,
        hspace=0.25, wspace=0.08
    )

    path_enhancement = os.path.join(
        folder_enhancement, nama_dasar + "_enhancement.png"
    )
    fig.savefig(path_enhancement, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Membuat dan menyimpan histogram setiap metode
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Histogram Citra\n" + nama_file,
        fontsize=16, fontweight="bold"
    )

    for ax, (citra, judul) in zip(axes.flat, daftar_citra):
        ax.hist(citra.ravel(), bins=256, range=[0, 256])
        ax.set_title(judul, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Intensitas Piksel")
        ax.set_ylabel("Frekuensi")
        ax.set_xlim(0, 255)
        ax.grid(alpha=0.2)

    fig.subplots_adjust(
        top=0.86, bottom=0.09, left=0.08, right=0.97,
        hspace=0.38, wspace=0.25
    )

    path_histogram = os.path.join(
        folder_histogram, nama_dasar + "_histogram.png"
    )
    fig.savefig(path_histogram, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Melakukan OCR pada keempat versi citra
    hasil_ocr = {}

    for citra, nama_metode in daftar_citra:
        teks = pytesseract.image_to_string(
            citra, config="--psm 7"
        ).strip()

        hasil_ocr[nama_metode] = (
            teks if teks else "[Tidak ada teks yang terdeteksi]"
        )

    # Menampilkan hasil OCR di terminal
    print(f"\nHasil pengujian OCR pada {nama_file}\n")

    for nomor_metode, (citra, nama_metode) in enumerate(
        daftar_citra, start=1
    ):
        print(
            f"{nomor_metode}. {nama_metode:<35} : "
            f"{hasil_ocr[nama_metode]}"
        )