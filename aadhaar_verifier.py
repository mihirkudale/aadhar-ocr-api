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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class PaddleAadhaarExtractor:
    def __init__(self, dpi=300):
        self.dpi = dpi
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def image_from_pdf(self, pdf_path):
        return convert_from_path(pdf_path, dpi=self.dpi)

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
        EXCLUDE_WORDS = ['dob', 'birth', 'male', 'female', 'government', 'uidai', 'year', 'india', 'authority', 'issue']

        for line in lines:
            l = line.lower()

            if re.search(r"^[a-zA-Z\s]{3,}$", line) and not any(w in l for w in EXCLUDE_WORDS):
                cleaned = re.sub(r"^(mr|ms|mrs)\.?\s*", "", line, flags=re.I).strip()
                name_candidates.append(cleaned)

            # Match any date pattern unless it includes issue-related keywords
            if re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", line):
                if not any(iss_kw in l for iss_kw in ["issue", "issued", "year of issue"]):
                    match = re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", line)
                    if match:
                        dob_candidates.append(match.group(1))

            # Gender
            if not gender:
                if 'male' in l:
                    gender = "Male"
                elif 'female' in l:
                    gender = "Female"
                elif 'transgender' in l:
                    gender = "Transgender"

            # Aadhaar
            if not aadhaar:
                digits = re.sub(r"\D", "", line)
                if len(digits) == 12:
                    aadhaar = digits

        logging.debug(f"[DEBUG] DOB Candidates: {dob_candidates}")

        # Pick best name match
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

        # Parse DOB
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
                lines = self.extract_text_lines(np.array(page))
                extracted = self.extract_fields(lines, record)
                if all(extracted.values()):
                    return extracted
            return extracted

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
    dob_match = dob_extracted == dob_record

    gender_extracted = extracted["Gender"].lower()
    gender_input = record.get("gender", "").lower()
    gender_match = gender_extracted == gender_input

    decoded_aadhaar = decode_base64_aadhaar(record.get("aadhar_number", ""))
    aadhaar_extracted = extracted["Aadhaar Number"]
    aadhaar_match = aadhaar_extracted == decoded_aadhaar

    all_present = all([extracted_name, gender_extracted, dob_extracted, aadhaar_extracted])
    all_match = all([name_match, dob_match, gender_match, aadhaar_match])

    decision = "ACCEPT" if all_present and all_match else "MANUAL_REVIEW"

    # logging.info("------ VERIFICATION DEBUG ------")
    # logging.info(f"Record Name       : '{full_name}'")
    # logging.info(f"Extracted Name    : '{extracted_name}' (Score: {name_score})")
    # logging.info(f"Record DOB        : '{raw_dob}' → Normalized: '{dob_record}'")
    # logging.info(f"Extracted DOB     : '{dob_extracted}'")
    # logging.info(f"Gender            : input='{gender_input}' vs extracted='{gender_extracted}' → {gender_match}")
    # logging.info(f"Decoded Aadhaar   : '{decoded_aadhaar}'")
    # logging.info(f"Extracted Aadhaar : '{aadhaar_extracted}'")
    # logging.info(f"Matches           : Name={name_match}, DOB={dob_match}, Gender={gender_match}, Aadhaar={aadhaar_match}")
    # logging.info(f"Final Decision    : {decision}")
    # logging.info("-------------------------------")

    return {
        "name_match": name_match,
        "dob_match": dob_match,
        "gender_match": gender_match,
        "aadhaar_match": aadhaar_match,
        "decision": decision
    }
