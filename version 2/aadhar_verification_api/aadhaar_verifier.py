import os
import re
import base64
import logging
import tempfile
import requests
import urllib3
import numpy as np
import cv2
from paddleocr import PaddleOCR
from datetime import datetime
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
import time

# Suppress insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PaddleAadhaarExtractor:
    def __init__(self, dpi=200):
        self.dpi = dpi
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.last_raw_ocr_result = []

    def image_from_pdf(self, pdf_path):
        return convert_from_path(pdf_path, dpi=self.dpi)

    def rotate_image(self, img, angle):
        if angle == 0:
            return img
        height, width = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        return cv2.warpAffine(img, matrix, (width, height), flags=cv2.INTER_LINEAR)

    def correct_image_rotation(self, image_np, confidence_threshold=60.0):
        result_0 = self.ocr.ocr(image_np, cls=True)
        self.last_raw_ocr_result = result_0
        lines_0 = [line[1][0] for block in result_0 for line in block if line[1][0].strip()]

        # Early extraction check
        extracted_0 = self.extract_fields(lines_0)
        # if all([extracted_0.get("Name"), extracted_0.get("DOB"), extracted_0.get("Gender"), extracted_0.get("Aadhaar Number")]):
        if sum(bool(extracted_0.get(k)) for k in ["Name", "DOB", "Gender", "Aadhaar Number"]) >= 3:
            logging.debug("[Rotation Skipped] All fields extracted correctly at 0°.")
            return image_np

        # Fallback to other angles
        best_orientation = image_np
        best_confidence = 0
        best_result = result_0

        for angle in [90, 180, 270]:
            rotated = self.rotate_image(image_np, angle)
            result = self.ocr.ocr(rotated, cls=True)
            confidences = [line[1][1] for block in result for line in block if line[1][0].strip()]
            avg_conf = np.mean(confidences) if confidences else 0
            if avg_conf > best_confidence:
                best_confidence = avg_conf
                best_orientation = rotated
                best_result = result

        self.last_raw_ocr_result = best_result
        return best_orientation

    def denoise_and_sharpen(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

    def extract_text_lines(self, image):
        result = self.ocr.ocr(image, cls=True)
        self.last_raw_ocr_result = result
        lines = [line[1][0] for block in result for line in block if line[1][0].strip()]
        for line in lines:
            logging.debug(f"[OCR LINE] {line}")
        return lines

    def extract_fields(self, lines, record=None):
        name_candidates = []
        dob_candidates = []
        gender = aadhaar = ""
        date_pattern = r"(\d{2}[/-]\d{2}[/-]\d{4})"

        for line in lines:
            l = line.lower()

            if re.fullmatch(r"[a-zA-Z\s]{3,}", line) and not any(w in l for w in ['dob', 'birth', 'male', 'female', 'government', 'uidai', 'year', 'india', 'authority', 'issue']):
                cleaned = re.sub(r"^(mr|ms|mrs|shri|smt)\.?\s*", "", line, flags=re.I).strip()
                name_candidates.append(cleaned)

            if re.search(date_pattern, l):
                if any(word in l for word in ["issue", "issued", "year of issue"]):
                    continue
                match = re.search(date_pattern, l)
                if match:
                    dob_candidates.append(match.group(1))

            if not gender:
                if 'male' in l:
                    gender = "Male"
                elif 'female' in l:
                    gender = "Female"
                elif 'transgender' in l:
                    gender = "Transgender"

            if not aadhaar:
                digits = re.sub(r"\D", "", line)
                if len(digits) == 12:
                    aadhaar = digits

        if not gender and lines:
            joined = " ".join(lines).lower()
            if "male" in joined:
                gender = "Male"
            elif "female" in joined:
                gender = "Female"

        expected_year = None
        if record:
            try:
                expected_year = int(record.get("dateOfbirth", "")[:4])
            except:
                pass

        if expected_year and dob_candidates:
            def year_diff(d):
                try:
                    return abs(expected_year - int(d[-4:]))
                except:
                    return 100
            dob_candidates = sorted(dob_candidates, key=year_diff)

        best_name = ""
        name_score = 0
        if record:
            full_name = f"{record.get('first_name', '')} {record.get('middle_name', '')} {record.get('last_name', '')}".strip().lower()
            full_name = re.sub(r"^(mr|ms|mrs|shri|smt)\.?\s*", "", full_name, flags=re.I)
            for candidate in name_candidates:
                score = max(
                    fuzz.token_set_ratio(candidate.lower(), full_name),
                    fuzz.partial_ratio(candidate.lower(), full_name)
                )
                if score > name_score:
                    best_name = candidate
                    name_score = score
        else:
            best_name = name_candidates[0] if name_candidates else ""

        parsed_dob = ""
        for dob in dob_candidates:
            try:
                parsed = datetime.strptime(dob.replace("/", "-"), "%d-%m-%Y").strftime("%Y-%m-%d")
                year = int(parsed.split("-")[0])
                if 1900 <= year <= datetime.now().year - 5:
                    parsed_dob = parsed
                    break
            except:
                continue

        return {
            "Name": best_name,
            "Gender": gender,
            "DOB": parsed_dob,
            "Aadhaar Number": aadhaar
        }

    def extract_from_file(self, url, record=None):
        temp_file_path = None
        start = time.time()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                response = requests.get(url, verify=False, timeout=30)
                response.raise_for_status()
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            images = self.image_from_pdf(temp_file_path)
            if not images:
                raise Exception("No image could be extracted from the PDF.")

            image = np.array(images[0])
            image = self.correct_image_rotation(image)
            # image = self.denoise_and_sharpen(image)
            lines = self.extract_text_lines(image)
            extracted = self.extract_fields(lines, record=record)
            logging.info(f"⏱️ Total time taken for {record.get('auth_id', 'N/A')}: {round(time.time() - start, 2)}s")
            return extracted
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

def decode_base64_aadhaar(encoded):
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except Exception:
        return ""

def normalize_dob(dob_str):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(dob_str, fmt).strftime("%Y-%m-%d")
        except:
            continue
    return ""

def verify_fields(extracted, record):
    def clean_name(name):
        return re.sub(r"^(mr|ms|mrs|shri|smt)\.?\s*", "", name.strip(), flags=re.I)

    full_name = clean_name(f"{record.get('first_name', '')} {record.get('middle_name', '')} {record.get('last_name', '')}").strip().lower()
    extracted_name = clean_name(extracted["Name"]).lower()

    name_score = max(
        fuzz.token_set_ratio(full_name, extracted_name),
        fuzz.partial_ratio(full_name, extracted_name)
    )
    name_match = name_score >= 68

    dob_record = normalize_dob(record.get("dateOfbirth", ""))
    dob_extracted = extracted["DOB"]
    dob_match = dob_extracted == dob_record

    gender_match = extracted["Gender"].lower() == record.get("gender", "").lower()

    decoded_aadhaar = decode_base64_aadhaar(record.get("aadhar_number", ""))
    aadhaar_match = extracted["Aadhaar Number"] == decoded_aadhaar

    all_present = all([extracted["Name"], extracted["Gender"], extracted["DOB"], extracted["Aadhaar Number"]])
    all_match = all([name_match, dob_match, gender_match, aadhaar_match])

    decision = "Verified" if all_present and all_match else "Not Verified - Manual Review"
    reasons = []
    if not name_match:
        reasons.append("name mismatch")
    if not dob_match:
        reasons.append("dob mismatch")
    if not gender_match:
        reasons.append("gender mismatch")
    if not aadhaar_match:
        reasons.append("aadhaar mismatch")

    return {
        "name_match": name_match,   
        "dob_match": dob_match,
        "gender_match": gender_match,
        "aadhaar_match": aadhaar_match,
        "decision": decision,
        "reason": ", ".join(reasons) if reasons else "all fields matched"
    }