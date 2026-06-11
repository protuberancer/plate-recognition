import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import re


replacements = {
    "0": "O", "8": "B", "2": "Z",
  }


def make_replacements(text: str) -> str:
    # Convert the recognized text to uppercase.
    text = text.upper()
    # Remove the substrings if it is present.
    text = text.translate(str.maketrans("", "", "RUS. "))
    letters = text[0] + text[4:6]
    numbers = text[1:4]
    region = text[6:] if len(text) > 6 else ''

    for k, v in replacements.items():
        letters = letters.replace(k, v)
        numbers = numbers.replace(v, k)
        region = region.replace(v, k)

    # Удаление лишних символов
    letters = re.sub(r'\d', '', letters)  # Удаляем цифры из буквенной части
    numbers = re.sub(r'[A-Z]', '', numbers)  # Удаляем буквы из числовой части
    region = re.sub(r'[A-Z]', '', region)  # Удаляем буквы из региона

    return letters[0] + numbers + letters[1:] + region


# Initialize the PaddleOCR model (downloads models if needed)
ocr = PaddleOCR(use_angle_cls=True, lang='en')
img_path = 'A001BP54.png'
result = ocr.ocr(img_path, cls=True)

if result and isinstance(result[0], list):
    detections = result[0]
else:
    detections = result

# Combine all recognized texts from the detections into one line.
recognized_text = ""
for detection in detections:
    # Skip if the detection is None
    if detection is None:
        continue

    # Ensure the detection has the expected structure (at least 2 elements)
    if isinstance(detection, (list, tuple)) and len(detection) >= 2 and detection[1]:
        recognized_text += detection[1][0]
        recognized_text = make_replacements(recognized_text)

if result and isinstance(result[0], list):
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
    # draw result
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/ttf/DejaVuSans.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(f'result_{recognized_text}.jpg')
print(f'plate:{recognized_text}')

