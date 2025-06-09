from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from aadhaar_verifier import PaddleAadhaarExtractor, verify_fields, decode_base64_aadhaar
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from werkzeug.utils import secure_filename
import logging
import numpy as np
import os
import time
import requests

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

client = MongoClient("mongodb://localhost:27017/")
db = client["aadhar"]
collection = db["coll_candidates"]
verification_collection = db["verification_results"]

extractor = PaddleAadhaarExtractor()

def generate_ref_number(decoded_aadhaar):
    url = 'https://aadhar.trti-maha.in:8080/'
    data = {
        'entered_uid': decoded_aadhaar,
        'entered_url': '',
        'entered_opr': 'struid'
    }
    try:
        response = requests.post(url, data=data, timeout=30, verify=False)
        response.raise_for_status()
        return response.json().get("refnum", "N/A")
    except Exception as e:
        logging.error(f"Error fetching Aadhaar ref number: {e}")
        return "N/A"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload-aadhaar", methods=["POST"])
def upload_and_extract():
    if 'aadhaar_pdf' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['aadhaar_pdf']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.abspath(os.path.join(app.config["UPLOAD_FOLDER"], filename)).replace("\\", "/")

    try:
        file.save(file_path)
        logging.info(f"✅ File uploaded: {filename}")

        partial_extracted = extractor.extract_from_file(file_path, record=None)
        aadhaar_number = partial_extracted.get("Aadhaar Number", "")
        if not aadhaar_number or len(aadhaar_number) != 12:
            return jsonify({"error": "Aadhaar number not detected or invalid"}), 422

        matched_record = None
        for candidate in collection.find({}):
            decoded = decode_base64_aadhaar(candidate.get("aadhar_number", ""))
            if decoded == aadhaar_number:
                matched_record = candidate
                break

        if not matched_record:
            return jsonify({"error": f"Aadhaar number {aadhaar_number} not found in DB"}), 404

        extracted = extractor.extract_from_file(file_path, matched_record)
        result = verify_fields(extracted, matched_record)

        try:
            raw_lines = extractor.last_raw_ocr_result
            confidences = [line[1][1] for block in raw_lines for line in block if line[1][0].strip()]
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        except:
            avg_confidence = 0.0

        decision = result.get("decision", "Manual_Review")
        status = "Verified" if decision.lower() == "accept" else "Not Verified"

        refnum = "N/A"
        if decision.lower() == "accept":
            decoded_uid = decode_base64_aadhaar(matched_record.get("aadhar_number", ""))
            refnum = generate_ref_number(decoded_uid)

        dob_raw = extracted.get("DOB", "")
        try:
            dob_formatted = datetime.strptime(dob_raw, "%Y-%m-%d").strftime("%d-%b-%Y")
        except:
            dob_formatted = dob_raw

        response_data = {
            "auth_id": matched_record.get("auth_id", "N/A"),
            "decision": decision,
            "status": status,
            "ocr_confidence": round(avg_confidence, 2),
            "extracted_name": extracted.get("Name", "N/A"),
            "extracted_dob": dob_formatted,
            "extracted_gender": extracted.get("Gender", "N/A"),
            "extracted_aadhaar": aadhaar_number,
            "aadhaar_refnum": refnum,
            "name_match": result["name_match"],
            "dob_match": result["dob_match"],
            "gender_match": result["gender_match"],
            "aadhaar_match": result["aadhaar_match"]
        }

        verification_collection.replace_one(
            {"decoded_aadhaar": aadhaar_number},
            {
                **response_data,
                "decoded_aadhaar": aadhaar_number,
                "timestamp": datetime.utcnow()
            },
            upsert=True
        )

        return jsonify(response_data), 200

    except Exception as e:
        logging.error(f"❌ Extraction failed: {e}")
        return jsonify({"error": f"Extraction failed: {e}"}), 500
    finally:
        time.sleep(0.5)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"🗑️ Deleted: {file_path}")
        except Exception as e:
            logging.warning(f"⚠️ Could not delete file: {file_path} — {e}")

def verify_single(record):
    try:
        aadhaar_url = f"https://cpetp.trti-maha.in/{record.get('aadhaar_doc', '')}"
        extracted = extractor.extract_from_file(aadhaar_url, record)
        record["decoded_aadhaar"] = decode_base64_aadhaar(record.get("aadhar_number", ""))
        result = verify_fields(extracted, record)

        result_entry = {
            "auth_id": record.get("auth_id"),
            "decoded_aadhaar": record["decoded_aadhaar"],
            "extracted_fields": extracted,
            "match_result": result,
            "timestamp": datetime.utcnow()
        }

        verification_collection.replace_one(
            {"decoded_aadhaar": record["decoded_aadhaar"]},
            result_entry,
            upsert=True
        )

        return result_entry

    except Exception as e:
        logging.error(f"Verification error: {e}")
        return None

@app.route("/run-batch-now", methods=["POST"])
def run_batch_now():
    logging.info("🟡 Manual batch verification triggered.")
    applicants = list(collection.find({}))
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(verify_single, record) for record in applicants]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    logging.info(f"✅ Batch verification done: {len(results)} records")
    return jsonify({"message": "Batch verification completed", "processed": len(results)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)