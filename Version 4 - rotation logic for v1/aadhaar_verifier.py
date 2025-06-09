import os
import re
import base64
import logging
import tempfile
import requests
import certifi
import urllib3
import numpy as np
import cv2
from paddleocr import PaddleOCR
from datetime import datetime
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
import threading

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class PaddleAadhaarExtractor:
    def __init__(self, dpi=150):
        self.dpi = dpi
        self._thread_local = threading.local()

    def get_ocr(self):
        if not hasattr(self._thread_local, "ocr"):
            self._thread_local.ocr = PaddleOCR(use_angle_cls=False, lang='en')
        return self._thread_local.ocr

    def image_from_pdf(self, pdf_path):
        return convert_from_path(pdf_path, dpi=self.dpi)

    def extract_text_lines(self, image):
        ocr = self.get_ocr()
        result = ocr.ocr(image, cls=False)
        self.last_raw_ocr_result = result
        lines = [(line[1][0], line[0][1]) for block in result for line in block if line[1][0].strip()]
        for line, _ in lines:
            logging.debug(f"[OCR LINE] {line}")
        return lines

    def extract_fields(self, lines, record=None):
        name_candidates = []
        dob_candidates = []
        gender = aadhaar = ""
        EXCLUDE_WORDS = ['dob', 'birth', 'male', 'female', 'government', 'uidai', 'year', 'india', 'authority', 'issue']

        for text, pos in lines:
            l = text.lower()

            if re.search(r"^[a-zA-Z\s]{3,}$", text) and not any(w in l for w in EXCLUDE_WORDS):
                cleaned = re.sub(r"^(mr|ms|mrs)\.?\s*", "", text, flags=re.I).strip()
                name_candidates.append(cleaned)

            if re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", text):
                if any(iss_kw in l for iss_kw in ["issue", "issued", "year of issue"]):
                    continue
                match = re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", text)
                if match:
                    x, y = pos
                    if x > 150:
                        dob_candidates.append((match.group(1), pos))

            if not gender:
                if 'male' in l:
                    gender = "Male"
                elif 'female' in l:
                    gender = "Female"
                elif 'transgender' in l:
                    gender = "Transgender"

            if not aadhaar:
                digits = re.sub(r"\D", "", text)
                if len(digits) == 12:
                    aadhaar = digits

        logging.debug(f"[DEBUG] DOB Candidates: {dob_candidates}")

        best_name = ""
        name_score = 0
        if record:
            full_name = f"{record.get('first_name', '')} {record.get('middle_name', '')} {record.get('last_name', '')}".strip().lower()
            for candidate in name_candidates:
                score = fuzz.token_set_ratio(candidate.lower(), full_name)
                if score > name_score:
                    best_name = candidate
                    name_score = score
        else:
            best_name = name_candidates[0] if name_candidates else ""

        parsed_dob = ""
        sorted_dobs = sorted(dob_candidates, key=lambda x: x[1])
        for dob, _ in sorted_dobs:
            try:
                parsed = datetime.strptime(dob.replace("/", "-"), "%d-%m-%Y").strftime("%Y-%m-%d")
                year = int(parsed.split("-")[0])
                if 1900 <= year <= datetime.now().year - 5:
                    parsed_dob = parsed
                    break
            except:
                continue

        logging.debug(f"[DEBUG] Parsed DOB: {parsed_dob}")

        return {
            "Name": best_name,
            "Gender": gender,
            "DOB": parsed_dob,
            "Aadhaar Number": aadhaar
        }

    def download_pdf_from_url(self, url):
        try:
            response = requests.get(url, verify=certifi.where(), timeout=10)
            response.raise_for_status()
        except requests.exceptions.SSLError:
            response = requests.get(url, verify=False, timeout=10)
        if response.status_code == 200:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
        raise Exception(f"Failed to download file: HTTP {response.status_code}")

    def extract_from_file(self, file_path_or_url, record=None):
        temp_pdf_path = None
        try:
            if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
                temp_pdf_path = self.download_pdf_from_url(file_path_or_url)
                file_path = temp_pdf_path
            else:
                file_path = file_path_or_url

            pages = self.image_from_pdf(file_path)

            for page in pages:
                img = np.array(page)

                # Step 1: Try 0° rotation first
                lines_0 = self.extract_text_lines(img)
                extracted_0 = self.extract_fields(lines_0, record)
                score_0 = sum([
                    bool(extracted_0["Name"]),
                    bool(extracted_0["DOB"]),
                    bool(extracted_0["Gender"]),
                    bool(extracted_0["Aadhaar Number"])
                ])
                if score_0 == 4:
                    logging.debug("✅ Skipping rotation: all fields found at 0°")
                    return extracted_0

                best_extracted = extracted_0
                best_score = score_0

                # Step 2: Try rotated versions
                for angle in [90, 180, 270]:
                    rotated = cv2.rotate(img, {
                        90: cv2.ROTATE_90_CLOCKWISE,
                        180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE
                    }[angle])
                    lines = self.extract_text_lines(rotated)
                    extracted = self.extract_fields(lines, record)
                    score = sum([
                        bool(extracted["Name"]),
                        bool(extracted["DOB"]),
                        bool(extracted["Gender"]),
                        bool(extracted["Aadhaar Number"])
                    ])
                    logging.debug(f"🔁 Rotation {angle}° → Score {score}")
                    if score > best_score:
                        best_score = score
                        best_extracted = extracted
                    if score == 4:
                        logging.debug(f"✅ All fields found at {angle}°")
                        break

                return best_extracted if best_extracted else {"Name": "", "Gender": "", "DOB": "", "Aadhaar Number": ""}

        except Exception as e:
            logging.error(f"Extraction failed: {e}")
            return {"Name": "", "Gender": "", "DOB": "", "Aadhaar Number": ""}
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

def decode_base64_aadhaar(encoded):
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except Exception:
        return ""

def normalize_dob(dob_str):
    try:
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y"):
            try:
                return datetime.strptime(dob_str, fmt).strftime("%Y-%m-%d")
            except:
                continue
    except:
        return ""
    return ""

def verify_fields(extracted, record):
    full_name = f"{record.get('first_name', '')} {record.get('middle_name', '')} {record.get('last_name', '')}".strip().lower()
    extracted_name = extracted["Name"].lower()

    name_score = max(
        fuzz.token_set_ratio(full_name, extracted_name),
        fuzz.ratio(full_name, extracted_name)
    )
    name_match = name_score >= 70

    raw_dob = record.get("dateOfbirth", "")
    dob_record = normalize_dob(raw_dob)
    dob_extracted = normalize_dob(extracted["DOB"])
    dob_match = dob_extracted == dob_record

    logging.debug(f"[DEBUG] DOB Record: {dob_record}")
    logging.debug(f"[DEBUG] DOB Extracted: {dob_extracted}")

    gender_extracted = extracted["Gender"].lower()
    gender_input = record.get("gender", "").lower()
    gender_match = gender_extracted == gender_input

    decoded_aadhaar = decode_base64_aadhaar(record.get("aadhar_number", ""))
    aadhaar_extracted = extracted["Aadhaar Number"]
    aadhaar_match = aadhaar_extracted == decoded_aadhaar

    all_present = all([extracted_name, gender_extracted, dob_extracted, aadhaar_extracted])
    all_match = all([name_match, dob_match, gender_match, aadhaar_match])

    decision = "Accept" if all_present and all_match else "Manual_Review"

    return {
        "name_match": name_match,
        "dob_match": dob_match,
        "gender_match": gender_match,
        "aadhaar_match": aadhaar_match,
        "decision": decision
    }