import base64
import json
import time
import logging
import sys
import psutil
import multiprocessing
from collections import Counter
from datetime import datetime
from pymongo import MongoClient, ASCENDING
import numpy as np
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from aadhaar_verifier import PaddleAadhaarExtractor, verify_fields, decode_base64_aadhaar

# Setup logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MongoDB setup (moved to global scope so subprocesses can access)
client = MongoClient("mongodb://localhost:27017/")
db = client["aadhar"]
collection = db["cpetp_db.tbl_candidate(police_military)"]
verification_collection = db["verification_results"]

# Auto-detect resources
cpu_cores = multiprocessing.cpu_count()
available_gb = psutil.virtual_memory().total / (1024 ** 3)

if available_gb >= 32:
    BATCH_SIZE = 1500
    max_workers = min(cpu_cores, 10)
elif available_gb >= 24:
    BATCH_SIZE = 1000
    max_workers = min(cpu_cores, 8)
elif available_gb >= 16:
    BATCH_SIZE = 750
    max_workers = min(cpu_cores, 6)
elif available_gb >= 12:
    BATCH_SIZE = 500
    max_workers = min(cpu_cores, 4)
elif available_gb >= 8:
    BATCH_SIZE = 300
    max_workers = min(cpu_cores, 3)
else:
    BATCH_SIZE = 100
    max_workers = 2

# CLI override
if len(sys.argv) >= 3:
    BATCH_SIZE = int(sys.argv[1])
    max_workers = int(sys.argv[2])

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
        json_data = response.json()
        return json_data.get("refnum", "N/A")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Aadhaar ref number: {str(e)}")
        return "Error"

def process_record(record_tuple):
    record, i, total = record_tuple
    extractor = PaddleAadhaarExtractor()
    start_time = time.time()

    try:
        aadhaar_path = record.get("aadhaar_doc")
        if not aadhaar_path:
            print(f"[{i+1}] {record.get('auth_id')} Missing Aadhaar path.")
            return None

        aadhaar_url = f"https://cpetp.trti-maha.in/{aadhaar_path}"
        for attempt in range(2):
            try:
                extracted = extractor.extract_from_file(aadhaar_url, record)
                break
            except Exception as e:
                if attempt == 1:
                    logging.error(f"[{i+1}] Final extraction failure: {e}")
                    return None
                time.sleep(1)

        decoded_aadhaar = decode_base64_aadhaar(record.get("aadhar_number", ""))
        record["decoded_aadhaar"] = decoded_aadhaar
        result = verify_fields(extracted, record)

        try:
            raw_lines = extractor.last_raw_ocr_result
            confidences = [line[1][1] for block in raw_lines for line in block if line[1][0].strip()]
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        except:
            avg_confidence = 0.0

        decision_raw = result["decision"]
        decision = decision_raw.strip().lower()
        aadhaar_status = "Verified" if decision == "accept" else "Not Verified"

        dob_raw = extracted.get("DOB", "")
        try:
            dob_formatted = datetime.strptime(dob_raw, "%Y-%m-%d").strftime("%d-%b-%Y")
        except:
            dob_formatted = dob_raw

        result_entry = {
            "auth_id": record.get("auth_id"),
            "decoded_aadhaar": decoded_aadhaar,
            "name_match": result["name_match"],
            "dob_match": result["dob_match"],
            "aadhaar_match": result["aadhaar_match"],
            "aadhaar_number": record.get("aadhar_number") or "",
            "decision": decision_raw,
            "aadhaar_status": aadhaar_status,
            "extracted_name": extracted.get("Name", ""),
            "extracted_dob": dob_formatted,
            "extracted_aadhaar": extracted.get("Aadhaar Number", ""),
            "ocr_confidence": round(avg_confidence, 2),
            "verified_by": 'Mihir',
            "verifier_role": 'AI'
        }

        if decision == "accept":
            refnum = "notavailable"
            if decoded_aadhaar:
                fetched_ref = generate_ref_number(decoded_aadhaar)
                if fetched_ref and fetched_ref not in ["N/A", "Error"]:
                    refnum = fetched_ref
                    print(f" Aadhaar Ref Number: {refnum}")
                else:
                    print(" Aadhaar reference number Not Available.")
            else:
                print(" Decoded Aadhaar number missing, setting refnum as 'notavailable'.")
            result_entry["aadhaar_ref_number"] = refnum

        print(
            f"[{i+1}/{total}] {record.get('auth_id')} → {decision_raw} | aadhaar_status={aadhaar_status} | "
            f"Name={result['name_match']} | DOB={result['dob_match']} | "
            f"Aadhaar={result['aadhaar_match']} | "
            f"Score={avg_confidence:.2f} | Extracted: {extracted['Name']}, {dob_formatted}, "
            f"{extracted['Aadhaar Number']}"
        )
        print(f" Time taken: {time.time() - start_time:.2f} seconds\n")

        # --- Immediate DB write ---
        if decision == "accept":
            base64_aadhaar = base64.b64encode(decoded_aadhaar.encode()).decode()
            name_parts = extracted.get("Name", "").strip().split()
            extracted_name_parts = (name_parts + ["", "", ""])[:3]

            collection.update_one(
                {"aadhar_number": base64_aadhaar},
                {"$set": {
                    "aadhaar_status": aadhaar_status,
                    "aadhaar_ref_number": result_entry.get("aadhaar_ref_number", ""),
                    "verified_by": result_entry.get("verified_by", "Mihir"),
                    "verifier_role": result_entry.get("verifier_role", "AI"),
                    "first_name": extracted_name_parts[0],
                    "middle_name": extracted_name_parts[1],
                    "last_name": extracted_name_parts[2],
                    "dateOfbirth": result_entry["extracted_dob"],
                    "aadhar_number": base64.b64encode(result_entry["extracted_aadhaar"].encode()).decode()
                }}
            )
        else:
            verification_collection.replace_one(
                {"decoded_aadhaar": decoded_aadhaar}, result_entry, upsert=True
            )

        return result_entry

    except Exception as e:
        logging.error(f"[{i+1}] {record.get('auth_id')} Error: {e}")
        return None

from tqdm import tqdm  # Make sure this is imported at the top

def run_batch_verification():
    start_batch = time.time()

    collection.create_index([("aadhar_number", ASCENDING)])
    verification_collection.create_index([("auth_id", ASCENDING)], unique=True)
    verification_collection.create_index([("decoded_aadhaar", ASCENDING)], unique=True)

    # total = min(2000, collection.count_documents({}))
    total = collection.count_documents({})
    logging.info(f"Total records to process: {total}")

    verified_total = 0
    manual_review_total = 0

    for skip in range(0, total, BATCH_SIZE):
        applicants = list(collection.find({
            "aadhaar_status": {"$exists": False}
        }).skip(skip).limit(BATCH_SIZE))

        logging.info(f"\n🔁 Processing batch {skip + 1} to {skip + len(applicants)}")

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_record, (record, i + skip, total)): i
                for i, record in enumerate(applicants)
            }

            # Add live progress bar for this batch
            for future in tqdm(as_completed(futures), total=len(futures), desc="🔄 Verifying", ncols=100):
                result = future.result()
                if result:
                    results.append(result)

        verified_count = sum(1 for r in results if r["decision"].strip().lower() == "accept")
        manual_review_count = len(results) - verified_count

        verified_total += verified_count
        manual_review_total += manual_review_count
        logging.info(f"✅ Batch completed: Verified={verified_count}, Manual={manual_review_count}")

    total_time_sec = time.time() - start_batch
    print(f"\n✅ All batches completed in {total_time_sec:.2f} seconds")
    print(f"🔢 Total records processed: {total}")
    print(f"✅ Verified records updated: {verified_total}")
    print(f"📁 Manual review saved: {manual_review_total}")


if __name__ == "__main__":
    print(f"\n🔧 Running with BATCH_SIZE = {BATCH_SIZE}, max_workers = {max_workers}")
    run_batch_verification()