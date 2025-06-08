import base64
import json
import time
import logging
from collections import Counter
from datetime import datetime
from pymongo import MongoClient, ASCENDING
import numpy as np
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from aadhaar_verifier import PaddleAadhaarExtractor, verify_fields, decode_base64_aadhaar

# Setup logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
        json_data = response.json()
        return json_data.get("refnum", "N/A")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Aadhaar ref number: {str(e)}")
        return "Error"

def process_record(record_tuple):
    record, i, total = record_tuple
    extractor = PaddleAadhaarExtractor()  # Each process gets its own instance
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
        status = "Verified" if decision == "accept" else "Not Verified"

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
            "gender_match": result["gender_match"],
            "aadhaar_match": result["aadhaar_match"],
            "decision": decision_raw,
            "status": status,
            "extracted_name": extracted.get("Name", ""),
            "extracted_gender": extracted.get("Gender", ""),
            "extracted_dob": dob_formatted,
            "extracted_aadhaar": extracted.get("Aadhaar Number", ""),
            "ocr_confidence": round(avg_confidence, 2)
        }

        if decision == "accept":
            refnum = "notavailable"
            if decoded_aadhaar:
                fetched_ref = generate_ref_number(decoded_aadhaar)
                if fetched_ref and fetched_ref not in ["N/A", "Error"]:
                    refnum = fetched_ref
                    print(f"\U0001F4CC Aadhaar Ref Number: {refnum}")
                else:
                    print("\u26A0\uFE0F Aadhaar reference number Not Available.")
            else:
                print("\u26A0\uFE0F Decoded Aadhaar number missing, setting refnum as 'notavailable'.")
            result_entry["aadhaar_refnum"] = refnum

        print(
            f"[{i+1}/{total}] {record.get('auth_id')} → {decision_raw} | Status={status} | "
            f"Name={result['name_match']} | DOB={result['dob_match']} | "
            f"Gender={result['gender_match']} | Aadhaar={result['aadhaar_match']} | "
            f"Score={avg_confidence:.2f} | Extracted: {extracted['Name']}, {dob_formatted}, "
            f"{extracted['Gender']}, {extracted['Aadhaar Number']}"
        )

        duration_sec = time.time() - start_time
        logging.info(f"[{i+1}/{total}] Processed {record.get('auth_id')} in {duration_sec:.2f} seconds")
        print(f"⏱️ Time taken: {duration_sec:.2f} seconds\n")
        return result_entry

    except Exception as e:
        logging.error(f"[{i+1}] {record.get('auth_id')} Error: {e}")
        return None

def run_batch_verification():
    start_batch = time.time()

    client = MongoClient("mongodb://localhost:27017/")
    db = client["aadhar"]
    collection = db["coll_candidates"]
    verification_collection = db["verification_results"]

    collection.create_index([("aadhar_number", ASCENDING)])
    verification_collection.create_index([("auth_id", ASCENDING)], unique=True)
    verification_collection.create_index([("decoded_aadhaar", ASCENDING)], unique=True)

    applicants = list(collection.find({}))
    total = len(applicants)
    logging.info(f"Total records fetched from MongoDB: {total}")

    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_record, (record, i, total)): i
            for i, record in enumerate(applicants)
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    if results:
        try:
            for r in results:
                verification_collection.replace_one(
                    {"decoded_aadhaar": r["decoded_aadhaar"]},
                    r,
                    upsert=True
                )
            logging.info(f"✅ Saved {len(results)} results to 'verification_results'")
        except Exception as e:
            logging.warning(f"⚠\uFE0F Error inserting results: {e}")
    else:
        logging.warning("❌ No results to insert.")

    total_time_sec = time.time() - start_batch
    logging.info(f"🗓️ Total time taken: {total_time_sec:.2f} seconds")
    print(f"\n🗓️ Total time taken: {total_time_sec:.2f} seconds")

    print("\nAccuracy Report")
    if results:
        decisions = [r['decision'].strip().lower() for r in results]
        counts = Counter(decisions)
        print(f"Verified (Accept)         : {counts.get('accept', 0)} ({counts.get('accept', 0)/len(results):.2%})")
        print(f"Not Verified (Manual Review): {counts.get('manual_review', 0)} ({counts.get('manual_review', 0)/len(results):.2%})")

        print("\nField-Level Accuracy")
        for field in ["name_match", "dob_match", "gender_match", "aadhaar_match"]:
            percent = sum(r[field] for r in results) / len(results)
            print(f"{field:<15}: {percent:.2%}")
    else:
        print("No records processed.")

if __name__ == "__main__":
    run_batch_verification()