import cv2
import numpy as np
import imutils
import pytesseract
from pdf2image import convert_from_path
import os

# Пути
pdf_path = "document.pdf"
poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === Функции ===
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# === Обработка ===
pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

full_text = ""

for i, page in enumerate(pages):
    print(f"[INFO] Обработка страницы {i+1}...")

    page_path = f"page_{i+1}.jpg"
    page.save(page_path, "JPEG")

    image = cv2.imread(page_path)
    orig = image.copy()
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2))
    else:
        print("[ВНИМАНИЕ] Контур не найден — используем оригинал")
        warped = orig

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_thresh = cv2.adaptiveThreshold(
        warped_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    text = pytesseract.image_to_string(warped_thresh, lang="eng+rus")
    full_text += f"\n--- Страница {i+1} ---\n" + text

    # Показать результат
    cv2.imshow(f"Страница {i+1}", imutils.resize(warped_thresh, height=600))
    cv2.waitKey(500)  # Пауза 0.5 сек

# === Сохраняем текст ===
with open("scanned_from_pdf.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("\n[✓] Готово. Распознанный текст сохранён в scanned_from_pdf.txt")
cv2.destroyAllWindows()
