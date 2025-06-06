import base64
import csv
import json
import logging
from collections import Counter
from pymongo import MongoClient
import numpy as np
from aadhaar_verifier import PaddleAadhaarExtractor, verify_fields, decode_base64_aadhaar
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Aadhaar extractor
extractor = PaddleAadhaarExtractor()


def generate_ref_number(decoded_aadhaar):
    """
    Calls the external service to fetch Aadhaar reference number.
    """
    url = 'https://aadhar.trti-maha.in:8080/'
    data = {
        'entered_uid': decoded_aadhaar,
        'entered_url': '',  # Optional: use actual request URL if needed
        'entered_opr': 'struid'
    }

    try:
        response = requests.post(url, data=data, timeout=30, verify=False)  # Insecure: SSL verification disabled
        response.raise_for_status()
        json_data = response.json()
        return json_data.get("refnum", "N/A")  # Return refnum or fallback
    except requests.exceptions.RequestException as e:
        logging.error(f" Error fetching Aadhaar ref number: {str(e)}")
        return "Error"

def run_batch_verification():
    # MongoDB connection
    client = MongoClient("mongodb://localhost:27017/")
    db = client["aadhar"]
    collection = db["coll_candidates"]

    applicants = list(collection.find({}))
    logging.info(f" Total records fetched from MongoDB: {len(applicants)}")

    results = []

    for i, record in enumerate(applicants):
        try:
            aadhaar_path = record.get("aadhaar_doc")
            if not aadhaar_path:
                print(f"[{i+1}] {record.get('auth_id')} Missing Aadhaar path.")
                continue

            aadhaar_url = f"https://cpetp.trti-maha.in/{aadhaar_path}"
            extracted = extractor.extract_from_file(aadhaar_url, record)

            # Decode Aadhaar number for comparison
            record["decoded_aadhaar"] = decode_base64_aadhaar(record.get("aadhar_number", ""))
            result = verify_fields(extracted, record)

            # OCR confidence score
            raw_lines = extractor.last_raw_ocr_result
            confidences = [line[1][1] for block in raw_lines for line in block if line[1][0].strip()]
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0

            result_entry = {
                "auth_id": record.get("auth_id"),
                "name_match": result["name_match"],
                "dob_match": result["dob_match"],
                "gender_match": result["gender_match"],
                "aadhaar_match": result["aadhaar_match"],
                "decision": result["decision"],
                "extracted_name": extracted["Name"],
                "extracted_gender": extracted["Gender"],
                "extracted_dob": extracted["DOB"],
                "extracted_aadhaar": extracted["Aadhaar Number"],
                "ocr_confidence": round(avg_confidence, 2)
            }

            

            # Clean one-line output
            print(
                f"[{i+1}/{len(applicants)}] {record.get('auth_id')} → {result['decision']} | "
                f"Name={result['name_match']} | DOB={result['dob_match']} | "
                f"Gender={result['gender_match']} | Aadhaar={result['aadhaar_match']} | "
                f"Score={avg_confidence:.2f} | "
                f"Extracted: {extracted['Name']}, {extracted['DOB']}, {extracted['Gender']}, { ['Aadhaar Number']}"
            )

            if result['decision'] == 'ACCEPT':
                decoded_aadhaar = record.get("decoded_aadhaar")
                if decoded_aadhaar:
                    refnum = generate_ref_number(decoded_aadhaar)
                    print(f"📎 Aadhaar Ref Number: {refnum}")
                    result_entry["aadhaar_refnum"] = refnum  # Optional: store in output

                    # ✅ Update MongoDB record with ref number and status
                    update_fields = {
                        "aadhaar_status": "verified",
                        "aadhaar_ref_number": refnum
                    }

                    collection.update_one(
                        {"_id": record["_id"]},
                        {"$set": update_fields}
                    )
                else:
                    print(" Decoded Aadhaar number missing, skipping refnum fetch.")


        except Exception as e:
            logging.error(f"[{i+1}] {record.get('auth_id')}Error: {e}")
            continue
            
        results.append(result_entry)
    # Save results
    if results:
        with open("aadhaar_verification_results.json", "w", encoding="utf-8") as f_json:
            json.dump(results, f_json, indent=2, ensure_ascii=False)

        with open("aadhaar_verification_results.csv", "w", newline='', encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    else:
        logging.warning(" No results to write. No records passed processing.")

    # Accuracy Summary
    print("\n Accuracy Report")
    if results:
        decisions = [r['decision'] for r in results]
        counts = Counter(decisions)
        total = len(results)

        print(f" ACCEPT         : {counts.get('ACCEPT', 0)} ({counts.get('ACCEPT', 0)/total:.2%})")
        print(f" MANUAL REVIEW  : {counts.get('MANUAL_REVIEW', 0)} ({counts.get('MANUAL_REVIEW', 0)/total:.2%})")

        print("\n Field-Level Accuracy")
        for field in ["name_match", "dob_match", "gender_match", "aadhaar_match"]:
            percent = sum(r[field] for r in results) / total
            print(f"{field:<15}: {percent:.2%}")
    else:
        print("No successful records processed.")

# Entry point
if __name__ == "__main__":
    run_batch_verification()
