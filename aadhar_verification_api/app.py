import os
import base64
import logging
import requests
import pprint
import certifi
import tempfile
import urllib3
from flask import Flask, request, jsonify
from pymongo import MongoClient
from aadhaar_verifier import PaddleAadhaarExtractor, verify_fields, decode_base64_aadhaar

# Setup
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
extractor = PaddleAadhaarExtractor()
API_SECRET = os.getenv("AADHAAR_API_SECRET", "trti2025")


@app.route("/verify-aadhaar", methods=["GET", "POST"])
def verify_aadhaar():
    logging.info("API called /verify-aadhaar")
    try:
        if request.method == "POST":
            applicants = request.get_json()

            # Validate input
            if not isinstance(applicants, list):
                return jsonify({"error": "Expected a list of applicant records"}), 400
        else:
            # GET: Fetch records from MongoDB
            client = MongoClient("mongodb://localhost:27017/")
            db = client["aadhar"]
            collection = db["15_Records"]
            applicants = list(collection.find({}, {"_id": 0}))
            logging.info(f"Fetched {len(applicants)} records from MongoDB")

        results = []

        for data in applicants:
            aadhaar_path = data.get("aadhaar_doc")

            # Validate Aadhaar path
            if not aadhaar_path or "://" in aadhaar_path or ".." in aadhaar_path:
                results.append({
                    "error": "Invalid Aadhaar document path",
                    "record": data.get("prn_number", "unknown")
                })
                continue

            aadhaar_url = f"https://cpetp.trti-maha.in/{aadhaar_path}"

            extracted = extractor.extract_from_file(aadhaar_url, data)
            verification_result = verify_fields(extracted, data)

            results.append({
                "prn_number": data.get("prn_number", "unknown"),
                "extracted_fields": extracted,
                "verification_result": verification_result
            })

        print("\n📋 Verification Results:\n")
        pprint.pprint(results)

        return jsonify(results)

    except Exception as e:
        logging.error(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
