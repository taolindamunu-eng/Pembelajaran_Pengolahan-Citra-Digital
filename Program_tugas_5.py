import os
import re
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

# Konfigurasi folder dan Tesseract OCR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(BASE_DIR, "gambar_5")
FOLDER_HASIL = os.path.join(BASE_DIR, "hasil_tugas_5")
GROUND_TRUTH = "Nomor ijazah: 571012022000056"
EKSTENSI_GAMBAR = ".jpg"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

os.makedirs(FOLDER_HASIL, exist_ok=True)

# Menghitung karakter benar dan akurasi OCR
def edit_distance(a, b):
    baris = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        baru = [i]

        for j, cb in enumerate(b, 1):
            baru.append(min(
                baru[j - 1] + 1,
                baris[j] + 1,
                baris[j - 1] + (ca != cb)
            ))

        baris = baru

    return baris[-1]


def hitung_akurasi(teks):
    teks = re.sub(r"\s+", " ", teks).strip()
    target = GROUND_TRUTH
    jarak = edit_distance(teks, target)
    benar = max(0, len(target) - jarak)
    akurasi = benar / len(target) * 100

    return benar, len(target), akurasi


# Melakukan OCR pada citra
def baca_ocr(citra):
    teks = pytesseract.image_to_string(
        citra, config="--psm 7"
    )
    return re.sub(r"\s+", " ", teks).strip()


# Menerapkan filtering, noise reduction, dan sharpening
def proses_gambar(gambar):
    gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    mean_filter = cv2.blur(gray, (3, 3))
    median_filter = cv2.medianBlur(gray, 3)
    gaussian_filter = cv2.GaussianBlur(gray, (3, 3), 0)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpening = cv2.filter2D(gray, -1, kernel)

    hasil = [
        ("ORIGINAL", cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)),
        ("GRAYSCALE", gray),
        ("MEAN FILTER", mean_filter),
        ("MEDIAN FILTER", median_filter),
        ("GAUSSIAN FILTER", gaussian_filter),
        ("SHARPENING", sharpening)
    ]

    return hasil


# Memeriksa folder gambar
if not os.path.isdir(FOLDER):
    raise FileNotFoundError(
        f"Folder gambar tidak ditemukan: {FOLDER}"
    )

daftar_gambar = sorted(
    nama for nama in os.listdir(FOLDER)
    if nama.lower().endswith(EKSTENSI_GAMBAR)
    and os.path.isfile(os.path.join(FOLDER, nama))
)

if not daftar_gambar:
    raise FileNotFoundError(
        f"Tidak ada gambar di folder: {FOLDER}"
    )

# Memproses seluruh gambar dan membandingkan hasil OCR
for nama_file in daftar_gambar:
    print(f"\nMemproses gambar: {nama_file}")

    path = os.path.join(FOLDER, nama_file)
    gambar = cv2.imread(path)

    if gambar is None:
        print(f"Gambar gagal dibaca: {nama_file}")
        continue

    hasil = proses_gambar(gambar)

    # Membuat gambar perbandingan enam tampilan
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(
        f"Filtering, Noise Reduction, dan Sharpening\n{nama_file}",
        fontsize=13,
        fontweight="bold"
    )

    for ax, (judul, citra) in zip(axes.flat, hasil):
        ax.imshow(citra, cmap="gray", vmin=0, vmax=255)
        ax.set_title(judul, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    nama_dasar = os.path.splitext(nama_file)[0]
    path_hasil = os.path.join(
        FOLDER_HASIL, f"{nama_dasar}_hasil.png"
    )

    fig.savefig(path_hasil, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Menjalankan OCR pada original dan setiap metode
    metode_ocr = [
        ("Original", hasil[0][1]),
        ("Mean Filter", hasil[2][1]),
        ("Median Filter", hasil[3][1]),
        ("Gaussian Filter", hasil[4][1]),
        ("Sharpening", hasil[5][1])
    ]

    hasil_ocr = []

    for nama_metode, citra in metode_ocr:
        teks = baca_ocr(citra)
        benar, total, akurasi = hitung_akurasi(teks)

        hasil_ocr.append((
            nama_metode,
            teks if teks else "[Tidak ada teks]",
            benar,
            total,
            akurasi
        ))

    # Menampilkan tabel OCR di terminal
    print(f"\nTABEL HASIL OCR - {nama_file}")
    print("-" * 100)
    print(
        f"{'Metode':<20}{'Hasil OCR':<38}"
        f"{'Karakter Benar':<20}{'Akurasi':<12}"
    )
    print("-" * 100)

    for metode, teks, benar, total, akurasi in hasil_ocr:
        print(
            f"{metode:<20}{teks[:36]:<38}"
            f"{benar}/{total:<16}{akurasi:.2f}%"
        )

    print("-" * 100)

# Ringkasan seluruh proses
print("\nSemua gambar selesai diproses.")
print("Jumlah gambar ditemukan:", len(daftar_gambar))
