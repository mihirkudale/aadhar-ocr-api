import os
import re
import base64
import logging
import tempfile
import requests
import certifi
import urllib3
import numpy as np
from paddleocr import PaddleOCR
from datetime import datetime
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
import threading

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class PaddleAadhaarExtractor:
    def __init__(self, dpi=150):  # Reduced DPI for faster processing
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
        result = ocr.ocr(image, cls=False)  # Angle classifier disabled for speed
        self.last_raw_ocr_result = result
        lines = [(line[1][0], line[0][1]) for block in result for line in block if line[1][0].strip()]
        for line, _ in lines:
            logging.debug(f"[OCR LINE] {line}")
        return lines

    def extract_fields(self, lines, record=None):
        name_candidates = []
        dob_candidates = []
        aadhaar = ""
        EXCLUDE_WORDS = ['dob', 'birth', 'male', 'female', 'government', 'uidai', 'year', 'india', 'authority', 'issue']

        for text, pos in lines:
            l = text.lower()

            if re.search(r"^[a-zA-Z\s]{3,}$", text) and not any(w in l for w in EXCLUDE_WORDS):
                cleaned = re.sub(r"^(mr|ms|mrs)\.?\s*", "", text, flags=re.I).strip()
                name_candidates.append(cleaned)

            if re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", text):
                if not any(iss_kw in l for iss_kw in ["issue", "issued", "year of issue"]):
                    match = re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", text)
                    if match:
                        dob_candidates.append((match.group(1), pos))

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

            try:
                pages = self.image_from_pdf(file_path)
            except Exception as e:
                logging.error(f"[PDF ERROR] Could not read PDF at {file_path}: {e}")
                return {"Name": "", "DOB": "", "Aadhaar Number": ""}

            for page in pages:
                try:
                    lines = self.extract_text_lines(np.array(page))
                    extracted = self.extract_fields(lines, record)
                    if all(extracted.values()):
                        return extracted
                except Exception as e:
                    logging.warning(f"[PAGE ERROR] Skipped a page due to: {e}")
                    continue

            return extracted if 'extracted' in locals() else {"Name": "", "DOB": "", "Aadhaar Number": ""}

        except Exception as e:
            logging.error(f"[EXTRACTION ERROR] {file_path_or_url} → {e}")
            return {"Name": "", "DOB": "", "Aadhaar Number": ""}

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
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(dob_str, fmt).strftime("%Y-%m-%d")
            except:
                continue
    except:
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
    dob_extracted = extracted["DOB"]

    # Enhanced DOB logic with year-only match fallback
    dob_match = False
    try:
        if dob_extracted == dob_record:
            dob_match = True
        else:
            record_year = datetime.strptime(dob_record, "%Y-%m-%d").year
            extracted_year = datetime.strptime(dob_extracted, "%Y-%m-%d").year
            if record_year == extracted_year:
                logging.info(f"[DOB MATCH] Accepted by year match: DB={dob_record}, OCR={dob_extracted}")
                dob_match = True
    except:
        pass

    decoded_aadhaar = decode_base64_aadhaar(record.get("aadhar_number", ""))
    aadhaar_extracted = extracted["Aadhaar Number"]
    aadhaar_match = aadhaar_extracted == decoded_aadhaar

    all_present = all([extracted_name, dob_extracted, aadhaar_extracted])
    all_match = all([name_match, dob_match, aadhaar_match])

    decision = "Accept" if all_present and all_match else "Manual_Review"

    return {
        "name_match": name_match,
        "dob_match": dob_match,
        "aadhaar_match": aadhaar_match,
        "decision": decision
    }
