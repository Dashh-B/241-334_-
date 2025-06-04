import cv2
import numpy as np
import imutils
import pytesseract
import os

# Путь к tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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

# === 1. Загрузка изображения ===
image_path = "document.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError("Файл не найден. Убедись, что 'document.jpg' в той же папке.")

image = cv2.imread(image_path)
orig = image.copy()
image = imutils.resize(image, height=500)

# === 2. Предобработка ===
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# === 3. Поиск документа по контуру ===
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

if screenCnt is None:
    raise Exception("Не удалось найти контур документа!")

# === 4. Перспективное преобразование ===
warped = four_point_transform(orig, screenCnt.reshape(4, 2))

# === 5. Бинаризация ===
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped_thresh = cv2.adaptiveThreshold(
    warped_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# === 6. OCR: распознавание текста ===
text = pytesseract.image_to_string(warped_thresh, lang="eng+rus")

# === 7. Сохранение текста ===
with open("scanned_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

# === 8. Отображение результатов ===
cv2.imshow("Оригинал", imutils.resize(orig, height=600))
cv2.imshow("Отсканировано", imutils.resize(warped_thresh, height=600))
cv2.imwrite("scanned_page_1.jpg", warped_thresh)

print("\n=== Распознанный текст ===\n")
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()
