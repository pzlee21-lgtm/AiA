"""
Flask REST API for SmartPharma RAG backend.
Main entry point that orchestrates retrieval and LLM generation.
"""

import time
import json
from flask import Flask, request, jsonify, make_response

from config import (
    TOP_K, OLLAMA_MODEL,
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    ALLOWED_SOURCES
)
from vector_store import build_or_load_store
from prompt_builder import make_prompt, ensure_verification_line, ANTICOAGULANT_DRUGS
from llm_client import call_ollama

# Initialize CORS if available
try:
    from flask_cors import CORS
    _HAS_CORS = True
except ImportError:
    _HAS_CORS = False

from reranker import hybrid_rerank

# =========================
# Initialize Flask app
# =========================
app = Flask(__name__)

if _HAS_CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def add_cors_headers(resp):
    """Ensure CORS headers are present on all responses."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


# =========================
# Load vector store
# =========================
print("Initializing vector store...")
STORE = build_or_load_store()
print(f"Vector store ready with {len(STORE.texts)} documents")


# =========================
# Routes
# =========================
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "docs": len(STORE.texts),
        "model": OLLAMA_MODEL
    })


@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    """
    Main RAG endpoint.
    Accepts POST with structured JSON.
    """
    # Handle preflight
    if request.method == "OPTIONS":
        resp = make_response("", 200)
        req_headers = request.headers.get("Access-Control-Request-Headers", "")
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = (
            req_headers if req_headers else "Content-Type, Authorization"
        )
        return resp

    t0 = time.time()

    # Parse request
    data = request.get_json(force=True, silent=True) or {}

    # Export received JSON to file for debugging
    with open('received_data.json', 'w') as json_file:
        json.dump(data, json_file, indent=2)

    print("\n" + "="*60)
    print("DEBUG: Received data from Flutter")
    print("="*60)
    print(json.dumps(data, indent=2))

    # Extract main question (required)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # Extract retrieval parameters
    k = int(data.get("k", TOP_K))
    age_raw = data.get("age", None) 
    try:
        age = int(age_raw) if age_raw else None
    except (ValueError, TypeError):
        age = None

    # Extract structured data
    patient_data = data.get("patient", {})
    labs_data = data.get("labs", {})
    vitals_data = data.get("vitals", {})
    diagnosis = (data.get("diagnosis") or "").strip()
    medicines = data.get("prescribed_medicines", [])

    print("\nDEBUG: Extracted fields")
    print(f"Question length: {len(question)}")
    print(f"Patient data: {patient_data}")
    print(f"Labs data: {labs_data}")
    print(f"Vitals data: {vitals_data}")
    print(f"Diagnosis: {diagnosis}")
    print(f"Medicines: {medicines}")
    print("="*60 + "\n")

    # Age check — warn if paediatric patient
    if age and age < 18:
        print("[WARN] Paediatric patient — only adult guidelines available")
    allowed = ALLOWED_SOURCES

    
    med_names = [m.get("name", "") for m in medicines if m.get("name")]

    # Separate antimicrobial vs anticoagulant drug names
    # Only use antimicrobial names in the rerank query to avoid diluting results
    antimicrobial_names = [
        name for name in med_names
        if name.lower() not in ANTICOAGULANT_DRUGS
    ]
    anticoagulant_names = [
        name for name in med_names
        if name.lower() in ANTICOAGULANT_DRUGS
    ]

    if anticoagulant_names:
        print(f"DEBUG: Anticoagulant drugs detected (excluded from rerank): {anticoagulant_names}")

    # Diagnosis only for initial retrieval — finds correct chapter first
    retrieval_query = diagnosis

    print(f"DEBUG: Retrieval query: {retrieval_query}")

    try:
        # Retrieve documents
        retrieved_all = STORE.search(retrieval_query, max(k * 3, k + 5))
        retrieved = [
            item for item in retrieved_all
            if item.get("meta", {}).get("source") in allowed
        ]

        # Fallback — return everything if filter yields nothing
        if not retrieved:
            retrieved = retrieved_all

        # Reranker uses diagnosis + ONLY antimicrobial drug names
        # Anticoagulant names are excluded to prevent diluting results
        rerank_query = f"{diagnosis} {' '.join(antimicrobial_names)} antimicrobial treatment"
        retrieved = hybrid_rerank(rerank_query, retrieved, top_k=k)

        print(f"DEBUG: Retrieved {len(retrieved)} documents")

        # =========================
        # Per-drug relevance check
        # =========================
        per_drug_results = {}
        for med in medicines:
            drug_name = med.get("name", "").strip()
            if not drug_name:
                continue

            # Search for each drug + diagnosis combination
            drug_query = f"{drug_name} {diagnosis}"
            drug_results = STORE.search(drug_query, k=3)

            # Filter to allowed sources
            drug_results = [
                item for item in drug_results
                if item.get("meta", {}).get("source") in allowed
            ]

            # Check if drug name actually appears in any retrieved chunk
            found_in_guidelines = any(
                drug_name.lower() in item["text"].lower()
                for item in drug_results
            )

            # Determine which source type the drug was found in
            source_type = None
            if found_in_guidelines:
                for item in drug_results:
                    if drug_name.lower() in item["text"].lower():
                        source_type = item.get("meta", {}).get("source")
                        break

            per_drug_results[drug_name] = {
                "found": found_in_guidelines,
                "source_type": source_type,
                "results": drug_results if found_in_guidelines else []
            }

            print(f"DEBUG: Drug '{drug_name}' — found={found_in_guidelines}, source={source_type}")

        # =========================
        # Generate answer with full context
        # =========================
        prompt = make_prompt(
            question=question,
            retrieved=retrieved,
            patient=patient_data,
            labs=labs_data,
            vitals=vitals_data,
            diagnosis=diagnosis,
            medicines=medicines,
            per_drug_results=per_drug_results
        )

        print(f"DEBUG: Generated prompt (full length {len(prompt)} chars):\n{prompt}\n")

        answer = call_ollama(prompt)

        print(f"\nDEBUG: Raw LLM response (length {len(answer)} chars):")
        print(f"'{answer}'")
        print("="*60)

        answer = ensure_verification_line(answer)

        print(f"DEBUG: After ensure_verification_line:")
        print(f"'{answer}'")
        print("="*60 + "\n")

        elapsed = round(time.time() - t0, 3)

        return jsonify({
            "question": question,
            "answer": answer,
            "retrieved": retrieved,
            "elapsed_s": elapsed
        })

    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        print(f"DEBUG: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Backend failed: {e}",
            "elapsed_s": elapsed
        }), 500


# =========================
# Main entry point
# =========================
if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)