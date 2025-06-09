from flask import Flask, request, jsonify
from flask_apscheduler import APScheduler
from aadhaar_verifier import PaddleAadhaarExtractor, verify_fields, decode_base64_aadhaar
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
import numpy as np
import time
import requests

app = Flask(__name__)
scheduler = APScheduler()
logging.getLogger("ppocr").setLevel(logging.ERROR)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["aadhar"]
collection = db["coll_candidates"]
verification_collection = db["verification_results"]

extractor = PaddleAadhaarExtractor()

class Config:
    SCHEDULER_API_ENABLED = True
app.config.from_object(Config())
scheduler.init_app(app)

@app.route("/")
def index():
    return "Aadhaar API with APScheduler (Windows safe) is running."

# 🧠 Verification logic
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

# 🕛 APScheduler job: daily at 00:00
@scheduler.task("cron", id="aadhaar_batch_daily", hour=0, minute=0)
def scheduled_verification():
    logging.info("🕐 Running scheduled Aadhaar verification")
    applicants = list(collection.find({}))
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(verify_single, record) for record in applicants]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    logging.info(f"✅ Finished scheduled verification — {len(results)} records processed")

# 🔘 Manual trigger
@app.route("/run-batch-now", methods=["POST"])
def run_batch_now():
    scheduled_verification()
    return jsonify({"message": "Manual batch verification triggered."}), 200

if __name__ == "__main__":
    scheduler.start()
    app.run(host="0.0.0.0", port=8000, debug=False)