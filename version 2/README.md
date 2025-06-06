# Aadhaar OCR Extraction API

This Flask API extracts Aadhaar details (Name, Gender, DOB, Aadhaar Number) from an Aadhaar PDF file URL using Paddle OCR.

## Features

* **POST API** to extract Aadhaar details from a PDF link
* Robust image preprocessing for better OCR
* Auto-cleanup of temporary files
* Returns JSON output, ready for integration

---

## Requirements

use

```
pip install -r requirements.txt
```
---

## Running the API

```bash
python batch_verifier.py (output will display in terminal)
```

The server will start at:
`http://localhost:5000`

---

## Usage(postman option)

### **POST** `/verify-aadhaar`

**Request**

* Content-Type: `application/json`
* Body:

  ```json
  {
      "pdf_url": "https://example.com/path/to/aadhaar.pdf"
  }
  ```

**Response**

* On success:

  ```json
  {
      "Name": "Shubham Avinash Sawant",
      "Gender": "Male",
      "DOB/Year of Birth": "15-Jun-1998",
      "Aadhaar Number": "123412341234"
  }
  ```
* On error:

  ```json
  {
      "error": "Description of the error"
  }
  ```

---

## Notes

* The accuracy depends on PDF quality. Noisy, low-res, or scanned PDFs may produce less accurate results.
* The API deletes the downloaded PDF after processing.
* Currently supports **URL input only**.

---

