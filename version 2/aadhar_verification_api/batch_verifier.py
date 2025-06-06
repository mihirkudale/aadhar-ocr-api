# from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from collections import Counter
from pymongo import MongoClient
import numpy as np
from aadhaar_verifier import PaddleAadhaarExtractor, verify_fields, decode_base64_aadhaar
import requests
import ssl

# Setup logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Disable SSL verification warning
requests.packages.urllib3.disable_warnings()

# Generate Aadhaar Reference Number
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
        logging.error(f"Error fetching Aadhaar ref number: {e}")
        return "Error"

# Process a single record
def process_record(index, total, record):
    extractor = PaddleAadhaarExtractor()
    start_time = time.time()
    result_entry = None

    try:
        aadhaar_path = record.get("aadhaar_doc")
        if not aadhaar_path:
            return f"[{index+1}] {record.get('auth_id')} Missing Aadhaar path.", None

        aadhaar_url = f"https://cpetp.trti-maha.in/{aadhaar_path}"
        extracted = extractor.extract_from_file(aadhaar_url, record)

        record["decoded_aadhaar"] = decode_base64_aadhaar(record.get("aadhar_number", ""))
        result = verify_fields(extracted, record)

        raw_lines = extractor.last_raw_ocr_result
        confidences = [line[1][1] for block in raw_lines for line in block if line[1][0].strip()]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        elapsed = round(time.time() - start_time, 2)

        result_entry = {
            "auth_id": record.get("auth_id"),
            "name_match": result["name_match"],
            "dob_match": result["dob_match"],
            "gender_match": result["gender_match"],
            "aadhaar_match": result["aadhaar_match"],
            "decision": result["decision"],
            "reason": result.get("reason", ""),
            "extracted_name": extracted["Name"],
            "extracted_gender": extracted["Gender"],
            "extracted_dob": extracted["DOB"],
            "extracted_aadhaar": extracted["Aadhaar Number"],
            "ocr_confidence": round(avg_confidence, 2)
        }

        output = (
            f"[{index+1}/{total}] {record.get('auth_id')} → {result['decision']} | "
            f"Name={result['name_match']} | DOB={result['dob_match']} | "
            f"Gender={result['gender_match']} | Aadhaar={result['aadhaar_match']} | "
            f"Score={avg_confidence:.2f} | Time={elapsed:.2f}s | "
            f"Reason: {result['reason']} | "
            f"Extracted: {extracted['Name']}, {extracted['DOB']}, {extracted['Gender']}, {extracted['Aadhaar Number']}"
        )

    except Exception as e:
        output = f"[{index+1}] {record.get('auth_id')} Error: {e}"

    return output, (record if result_entry and result_entry["decision"] == "Verified" else None, result_entry)

# Batch processor
def run_batch_verification():
    start_time = time.time()  # Start timer ✅

    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    db = client["aadhar"]
    collection = db["coll_candidates"]
    result_collection = db["aadhaar_verification_results"]

    try:
        collection.create_index([("aadhar_number", 1)], unique=True)
        collection.create_index([("auth_id", 1)], unique=True)
    except Exception as e:
        logging.warning(f"Index creation warning: {e}")

    applicants = list(collection.find({}))
    logging.info(f"Total records fetched: {len(applicants)}")

    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_record, i, len(applicants), record) for i, record in enumerate(applicants)]

        for future in as_completed(futures):
            output, data = future.result()
            print(output)

            if data:
                record, result_entry = data
                if record and result_entry:
                    decoded_aadhaar = record.get("decoded_aadhaar")
                    if decoded_aadhaar:
                        refnum = generate_ref_number(decoded_aadhaar)
                        print(f"📌 Aadhaar Ref Number: {refnum}")
                        result_entry["aadhaar_refnum"] = refnum
                        collection.update_one(
                            {"_id": record["_id"]},
                            {"$set": {
                                "aadhaar_status": "verified",
                                "aadhaar_ref_number": refnum
                            }}
                        )
                if result_entry:
                    results.append(result_entry)

    # Store results
    if results:
        try:
            result_collection.insert_many(results)
            logging.info(f"✅ {len(results)} results inserted into results collection.")
        except Exception as e:
            logging.error(f"❌ Insertion failed: {e}")
    else:
        logging.warning("No records passed verification.")

    # Accuracy summary
    print("\nAccuracy Report")
    if results:
        decisions = [r['decision'] for r in results]
        counts = Counter(decisions)
        total = len(results)

        print(f"Verified               : {counts.get('Verified', 0)} ({counts.get('Verified', 0)/total:.2%})")
        print(f"Not Verified - Review  : {counts.get('Not Verified - Manual Review', 0)} ({counts.get('Not Verified - Manual Review', 0)/total:.2%})")

        print("\nField-Level Accuracy")
        for field in ["name_match", "dob_match", "gender_match", "aadhaar_match"]:
            percent = sum(r[field] for r in results) / total
            print(f"{field:<20}: {percent:.2%}")
    else:
        print("No successful records processed.")

    # ✅ Print total time
    print(f"\n🕒 Total time: {round(time.time() - start_time, 2)} seconds")


if __name__ == "__main__":
    run_batch_verification()
