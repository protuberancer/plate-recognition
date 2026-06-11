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

# Folder containing test images
test_folder = 'test'

# Counters for statistics
total_images = 0
mismatch_count = 0
mismatches = []  # To store details about each mismatch

# Loop over all files in the test folder
for filename in os.listdir(test_folder):
    # Process only image files (adjust extensions as needed)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    total_images += 1

    # Expected text is taken from the file name (without extension)
    expected_text = os.path.splitext(filename)[0]

    # Build the complete image path
    img_path = os.path.join(test_folder, filename)

    # Run OCR on the image.
    # Note: For many images, PaddleOCR returns a nested list: a list with one element
    # that is the list of detection results.
    result = ocr.ocr(img_path, cls=True)

    # Depending on the image, the detections may be nested.
    # If result[0] is a list of detections, use it; otherwise, use result directly.
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

    # Compare the recognized text to the expected text
    if recognized_text != expected_text:
        mismatch_count += 1
        mismatches.append((filename, expected_text, recognized_text))
        print(f"Mismatch in file '{filename}': expected '{expected_text}', got '{recognized_text}'")
    else:
        print(f"Match in file '{filename}': recognized '{recognized_text}'")

# Output overall statistics
print("\n--- Statistics ---")
print(f"Total images processed: {total_images}")
print(f"Matches: {total_images - mismatch_count}")
print(f"Mismatches: {mismatch_count}")
if total_images > 0:
    mismatch_percentage = (mismatch_count / total_images) * 100
    print(f"Mismatch Percentage: {mismatch_percentage:.2f}%")

# (Optional) Save drawn results for one of the mismatched images
# This block is just an example to show how to draw and save the OCR results.
if mismatches:
    sample_file, exp, rec = mismatches[0]
    sample_path = os.path.join(test_folder, sample_file)
    result_sample = ocr.ocr(sample_path, cls=True)
    if result_sample and isinstance(result_sample[0], list):
        sample_detections = result_sample[0]
    else:
        sample_detections = result_sample

    # Open the image
    image = Image.open(sample_path).convert('RGB')
    # Extract boxes, texts, and scores for drawing
    boxes = [line[0] for line in sample_detections]
    txts = [line[1][0] for line in sample_detections]
    scores = [line[1][1] for line in sample_detections]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/ttf/DejaVuSans.ttf')
    im_show = Image.fromarray(im_show)
    drawn_result_path = os.path.join('./', "drawn_" + sample_file)
    im_show.save(drawn_result_path)
    print(f"Saved drawn OCR result for sample mismatch to '{drawn_result_path}'")
