
import cv2
import os
import re
import numpy as np
import pytesseract
from pytesseract import Output

FOLDER = "gambar_7"
OUTPUT = "hasil_tugas_7"
GROUND_TRUTH = "571012022000056"
JARAK = 20

# Lokasi Tesseract OCR
for path in [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break
else:
    raise SystemExit("ERROR: Tesseract tidak ditemukan!")

OCR_DIR = os.path.join(OUTPUT, "enhancement_ocr")
SIG_DIR = os.path.join(OUTPUT, "deteksi_tanda_tangan")
os.makedirs(OCR_DIR, exist_ok=True)
os.makedirs(SIG_DIR, exist_ok=True)


# 1. ENHANCEMENT

def contrast_stretch(img):
    mn = float(np.min(img))
    mx = float(np.max(img))

    if mx == mn:
        return img.copy()

    img_float = img.astype(np.float32)

    hasil = (img_float - mn) * 255.0 / (mx - mn)

    return np.clip(hasil, 0, 255).astype(np.uint8)

def clahe(img):
    return cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(img)


def gamma_correction(img, gamma=1.5):
    table = np.array([
        ((i / 255.0) ** (1 / gamma)) * 255
        for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(img, table)


# 2. OCR DAN CER

def ekstrak_nomor(text):
    kandidat = re.findall(r"\d{10,}", text.replace(" ", ""))
    if not kandidat:
        return ""
    return min(
        kandidat,
        key=lambda x: abs(len(x) - len(GROUND_TRUTH))
    )


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                curr[-1] + 1, prev[j] + 1,
                prev[j - 1] + cost
            ))
        prev = curr
    return prev[-1]


def hitung_cer(prediksi):
    if not prediksi:
        return 1.0
    return levenshtein(
        prediksi, GROUND_TRUTH
    ) / len(GROUND_TRUTH)


# 3. DETEKSI TEKS DENGAN OCR

def buat_text_mask(gray):
    data = pytesseract.image_to_data(
        gray, config="--psm 6",
        output_type=Output.DICT
    )
    mask = np.zeros(gray.shape, dtype=np.uint8)
    visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = 0

        if conf < 20:
            continue

        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]
        p = 5
        x1, y1 = max(0, x-p), max(0, y-p)
        x2 = min(gray.shape[1], x+w+p)
        y2 = min(gray.shape[0], y+h+p)
        mask[y1:y2, x1:x2] = 255
        cv2.rectangle(
            visual, (x, y), (x+w, y+h), (0, 0, 255), 2
        )

    return mask, visual


# 4. THRESHOLDING DAN MORFOLOGI

def deteksi_metode(gray, text_mask, adaptive=False):
    if adaptive:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 15
        )
    else:
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    binary[text_mask == 255] = 0
    threshold_img = binary.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    opening_img = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel, iterations=1
    )
    closing_img = cv2.morphologyEx(
        opening_img, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        closing_img, 8
    )
    h, w = gray.shape
    total = h * w
    candidates = []

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        if area < total * 0.0003 or bw < w * 0.08:
            continue

        aspect = bw / max(bh, 1)
        density = area / max(bw * bh, 1)
        score = 0

        if bw >= w * 0.15:
            score += 2
        if aspect >= 2:
            score += 2
        if area >= total * 0.001:
            score += 1
        if density < 0.50:
            score += 1
        if score >= 4:
            candidates.append((area, score))

    if not candidates:
        result = ("ABSENT", 0.0, 0)
    else:
        area, score = max(
            candidates, key=lambda x: (x[1], x[0])
        )
        result = ("PRESENT", area / total * 100, score)

    return (
        *result, threshold_img, opening_img, closing_img
    )


def deteksi_tanda_tangan(gray):
    mask, ocr_visual = buat_text_mask(gray)

    # Global Thresholding
    _, global_img = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY_INV
    )
    global_img[mask == 255] = 0

    otsu = deteksi_metode(gray, mask)
    adaptive = deteksi_metode(gray, mask, True)

    if adaptive[2] > otsu[2] or (
        adaptive[2] == otsu[2] and adaptive[1] > otsu[1]
    ):
        chosen, nama = adaptive, "Adaptive"
    else:
        chosen, nama = otsu, "Otsu"

    return (
        chosen[0], chosen[1], nama, ocr_visual,
        global_img, otsu[3], adaptive[3],
        chosen[4], chosen[5]
    )


# 5. PANEL VISUALISASI

def buat_panel(img, judul, ukuran=(320, 220)):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    scale = min(ukuran[0] / w, ukuran[1] / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    kecil = cv2.resize(img, (nw, nh))

    panel = np.full(
        (ukuran[1] + 35, ukuran[0], 3),
        255, dtype=np.uint8
    )
    x = (ukuran[0] - nw) // 2
    y = 35 + (ukuran[1] - nh) // 2
    panel[y:y+nh, x:x+nw] = kecil

    cv2.putText(
        panel, judul[:40], (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1
    )
    return panel


def buat_ringkasan(nomor, cer, metode):
    panel = np.full((255, 320, 3), 255, dtype=np.uint8)
    teks = [
        "HASIL OCR",
        "Nomor: " + (nomor or "Tidak terbaca"),
        f"CER: {cer*100:.2f}%",
        "Enhancement terbaik:",
        metode
    ]

    for i, baris in enumerate(teks):
        cv2.putText(
            panel, baris[:38], (10, 35+i*40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1
        )
    return panel


# Jarak putih berlaku untuk KEDUA kolase
def simpan_kolase(panels, path, kolom=3, jarak=JARAK):
    panels = [
        buat_panel(img, judul) for judul, img in panels
    ]

    tinggi, lebar = panels[0].shape[:2]
    baris = (len(panels) + kolom - 1) // kolom
    tinggi_total = baris * tinggi + (baris + 1) * jarak
    lebar_total = kolom * lebar + (kolom + 1) * jarak

    kanvas = np.full(
        (tinggi_total, lebar_total, 3),
        255, dtype=np.uint8
    )

    for i, panel in enumerate(panels):
        y = jarak + (i // kolom) * (tinggi + jarak)
        x = jarak + (i % kolom) * (lebar + jarak)
        kanvas[y:y+tinggi, x:x+lebar] = panel

    cv2.imwrite(path, kanvas)


# 6. PEMROSESAN SEMUA GAMBAR

if not os.path.isdir(FOLDER):
    raise SystemExit("Folder 'gambar' tidak ditemukan!")

files = sorted([
    f for f in os.listdir(FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not files:
    raise SystemExit("Tidak ada gambar dalam folder 'gambar'!")

for filename in files:
    img = cv2.imread(os.path.join(FOLDER, filename))
    if img is None:
        print("Gambar gagal dibaca:", filename)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    metode = {
        "Contrast Stretching": contrast_stretch(gray),
        "CLAHE": clahe(gray),
        "Gamma Correction": gamma_correction(gray)
    }

    hasil_ocr = []
    for nama, enhanced in metode.items():
        text = pytesseract.image_to_string(
            enhanced, config="--psm 6"
        )
        nomor = ekstrak_nomor(text)
        hasil_ocr.append((
            nama, nomor, hitung_cer(nomor), enhanced
        ))
    
    terbaik = min(hasil_ocr, key=lambda x: x[2])

    (signature, percentage, metode_sig, ocr_visual,
     global_img, otsu_img, adaptive_img,
     opening_img, closing_img) = deteksi_tanda_tangan(gray)

    nama_file = os.path.splitext(filename)[0]

    # Kolase enhancement dan OCR
    panels_ocr = [
        ("Citra asli", img),
        ("Grayscale", gray),
        ("Contrast Stretching", metode["Contrast Stretching"]),
        ("CLAHE", metode["CLAHE"]),
        ("Gamma Correction", metode["Gamma Correction"]),
        ("Ringkasan OCR", buat_ringkasan(
            terbaik[1], terbaik[2], terbaik[0]
        ))
    ]
    simpan_kolase(
        panels_ocr,
        os.path.join(OCR_DIR, nama_file + "_ocr.jpg"),
        kolom=3,
        jarak=JARAK
    )

    # Kolase deteksi tanda tangan
    panels_sig = [
        ("Citra asli", img),
        ("Grayscale", gray),
        ("Deteksi teks OCR", ocr_visual),
        ("Global Thresholding", global_img),
        ("Otsu Thresholding", otsu_img),
        ("Adaptive Thresholding", adaptive_img),
        ("Morphological Opening", opening_img),
        ("Morphological Closing", closing_img)
    ]
    simpan_kolase(
    panels_sig,
    os.path.join(SIG_DIR, nama_file + "_deteksi.jpg"),
    kolom=4,
    jarak=JARAK
)

    # Satu blok hasil terminal untuk setiap gambar
    print("\n" + "=" * 55)
    print(f"{'Nama gambar':<24}: {filename}")
    print(f"{'Hasil OCR':<24}: {terbaik[1] or 'Tidak terbaca'}")
    print(f"{'CER':<24}: {terbaik[2]*100:.2f}%")
    print(f"{'Enhancement terbaik':<24}: {terbaik[0]}")
    print(f"{'Tanda tangan':<24}: {signature}")
    print("=" * 55)

print("\nSemua gambar selesai diproses.")
print("Hasil visualisasi tersimpan di folder:", OUTPUT)