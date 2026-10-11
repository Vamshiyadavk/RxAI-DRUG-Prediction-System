"""
RxAI — Flask Backend
────────────────────
1.  pip install -r requirements.txt
2.  python app.py  (auto-trains model if drug_model.pkl is missing)
3.  Open http://localhost:5000 in your browser
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os
import sys

# Ensure Windows console supports UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE, "drug_model.pkl")

if not os.path.exists(model_path):
    print("drug_model.pkl not found. Training model now...")
    try:
        import train_model
        train_model.main()
    except Exception as exc:
        print(f"Error during training: {exc}")

bundle = joblib.load(model_path)

model       = bundle["model"]
le_sex      = bundle["le_sex"]
le_bp       = bundle["le_bp"]
le_chol     = bundle["le_chol"]
le_drug     = bundle["le_drug"]
drug_lookup = bundle["drug_lookup"]
test_acc    = bundle["test_accuracy"]
cv_acc      = bundle["cv_accuracy"]
n_classes   = bundle["n_classes"]
n_trees     = bundle["n_trees"]


# ── Allow CORS from browser ────────────────────────────────────
@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(BASE, "index.html")


@app.route("/<path:path>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def handle_options(path=""):
    return jsonify({}), 200


# ── GET /health ────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "RandomForestClassifier",
        "n_classes": n_classes,
        "test_accuracy": round(test_acc * 100, 2),
        "cv_accuracy": round(cv_acc * 100, 2),
    })


# ── GET /presets ───────────────────────────────────────────────
@app.route("/presets", methods=["GET"])
def get_presets():
    presets = [
        {
            "id": "hypertensive-elderly-male",
            "title": "Hypertensive Elderly Male",
            "tag": "Cardiology",
            "age": 65,
            "sex": "M",
            "bp": "HIGH",
            "cholesterol": "HIGH",
            "na_to_k": 13.8,
            "notes": "Co-existing hypertension and hyperlipidemia with borderline potassium."
        },
        {
            "id": "young-active-female",
            "title": "Young Active Female",
            "tag": "General Health",
            "age": 24,
            "sex": "F",
            "bp": "LOW",
            "cholesterol": "NORMAL",
            "na_to_k": 28.5,
            "notes": "Mild hypotension with elevated Na/K ratio consistent with high fluid turnover."
        },
        {
            "id": "metabolic-risk-male",
            "title": "Middle-Aged Metabolic Risk",
            "tag": "Endocrinology",
            "age": 52,
            "sex": "M",
            "bp": "HIGH",
            "cholesterol": "NORMAL",
            "na_to_k": 9.4,
            "notes": "Hypertension with low Na/K ratio requiring careful diuretic/electrolyte evaluation."
        },
        {
            "id": "post-menopausal-cardiac",
            "title": "Post-Menopausal Cardiac Risk",
            "tag": "Preventive",
            "age": 70,
            "sex": "F",
            "bp": "NORMAL",
            "cholesterol": "HIGH",
            "na_to_k": 17.2,
            "notes": "Normal blood pressure but significantly elevated cholesterol in geriatric female."
        },
        {
            "id": "acute-dehydration-electrolyte",
            "title": "Electrolyte Imbalance Profile",
            "tag": "Emergency/Fluids",
            "age": 38,
            "sex": "M",
            "bp": "LOW",
            "cholesterol": "NORMAL",
            "na_to_k": 34.2,
            "notes": "Markedly elevated Na/K ratio with systemic hypotension indicating severe volume depletion."
        }
    ]
    return jsonify(presets)


# ── GET /drugs ─────────────────────────────────────────────────
@app.route("/drugs", methods=["GET"])
def list_drugs():
    all_drugs = []
    for dname in sorted(drug_lookup.keys()):
        dinfo = drug_lookup.get(dname, {})
        all_drugs.append({
            "name": dname,
            "drug_class": dinfo.get("drug_class", "Therapeutic Agent"),
            "indication": dinfo.get("indication", ""),
            "description": dinfo.get("description", "")
        })
    return jsonify({
        "total": len(all_drugs),
        "drugs": all_drugs
    })


# ── POST /predict ──────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        d    = request.get_json(force=True)
        age  = float(d["age"])
        sex  = str(d["sex"]).upper().strip()
        bp   = str(d["bp"]).upper().strip()
        chol = str(d["cholesterol"]).upper().strip()
        nak  = float(d["na_to_k"])

        # Basic validation
        if not (1 <= age <= 120):
            return jsonify({"error": "Age must be between 1 and 120"}), 400
        if sex not in le_sex.classes_:
            return jsonify({"error": f"Sex must be 'M' or 'F'"}), 400
        if bp not in le_bp.classes_:
            return jsonify({"error": "BP must be HIGH, NORMAL, or LOW"}), 400
        if chol not in le_chol.classes_:
            return jsonify({"error": "Cholesterol must be HIGH or NORMAL"}), 400
        if not (6.0 <= nak <= 38.5):
            return jsonify({"error": "Na/K ratio must be between 6.0 and 38.5"}), 400

        X = np.array([[
            age,
            le_sex.transform([sex])[0],
            le_bp.transform([bp])[0],
            le_chol.transform([chol])[0],
            nak,
        ]])

        pred_enc   = model.predict(X)[0]
        proba      = model.predict_proba(X)[0]
        drug_name  = le_drug.inverse_transform([pred_enc])[0]
        confidence = float(np.max(proba)) * 100
        info       = drug_lookup.get(drug_name, {})

        # Top 3 alternatives
        top3 = []
        for idx in np.argsort(proba)[::-1][:3]:
            dname = le_drug.inverse_transform([idx])[0]
            dinfo = drug_lookup.get(dname, {})
            top3.append({
                "drug"       : dname,
                "drug_class" : dinfo.get("drug_class", ""),
                "confidence" : round(float(proba[idx]) * 100, 1),
            })

        # Generate intelligent clinical alerts based on patient parameters & medication
        alerts = []
        if bp == "HIGH" and chol == "HIGH":
            alerts.append({
                "severity": "danger",
                "badge": "High Risk",
                "title": "Dual Cardiovascular Risk Profile",
                "message": "Patient has concurrent hypertension and hypercholesterolemia, substantially increasing 10-year ASCVD risk."
            })
        elif bp == "HIGH":
            alerts.append({
                "severity": "warning",
                "badge": "Monitoring",
                "title": "Hypertension Detected",
                "message": "Routine ambulatory BP monitoring and sodium intake restriction recommended."
            })
        elif bp == "LOW":
            alerts.append({
                "severity": "warning",
                "badge": "Precaution",
                "title": "Hypotension Alert",
                "message": "Screen for postural dizziness, autonomic neuropathy, or hypovolemic state."
            })

        if nak < 10.0:
            alerts.append({
                "severity": "warning",
                "badge": "Electrolytes",
                "title": "Low Na/K Ratio (< 10.0)",
                "message": "Possible hypokalemia or excessive sodium loss. Check basic metabolic panel."
            })
        elif nak > 25.0:
            alerts.append({
                "severity": "info",
                "badge": "Biomarker",
                "title": "High Na/K Ratio (> 25.0)",
                "message": "Elevated ratio indicative of high potassium turnover or volume loss. Maintain adequate hydration."
            })

        if age >= 65:
            alerts.append({
                "severity": "info",
                "badge": "Geriatric",
                "title": "Geriatric Pharmacotherapy",
                "message": "Reduced renal drug clearance is common in patients ≥ 65. Follow 'start-low, go-slow' principle."
            })

        if sex == "F" and any(ace in drug_name for ace in ["Lisinopril", "Enalapril", "Ramipril"]):
            alerts.append({
                "severity": "warning",
                "badge": "Contraindication",
                "title": "Teratogenicity Precaution",
                "message": "ACE inhibitors are strictly contraindicated during pregnancy due to fetotoxicity risks. Verify pregnancy status."
            })

        return jsonify({
            "drug"        : drug_name,
            "drug_class"  : info.get("drug_class", "Therapeutic Agent"),
            "indication"  : info.get("indication", ""),
            "description" : info.get("description", ""),
            "confidence"  : round(confidence, 1),
            "alternatives": top3,
            "clinical_alerts": alerts,
            "patient_summary": {
                "age": age,
                "sex": "Male" if sex == "M" else "Female",
                "bp": bp,
                "cholesterol": chol,
                "na_to_k": nak
            },
            "model_info"  : {
                "test_accuracy": round(test_acc * 100, 2),
                "cv_accuracy"  : round(cv_acc   * 100, 2),
                "n_trees"      : n_trees,
                "n_classes"    : n_classes,
            },
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── GET /model-info ────────────────────────────────────────────
@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "test_accuracy": round(test_acc * 100, 2),
        "cv_accuracy"  : round(cv_acc   * 100, 2),
        "n_trees"      : n_trees,
        "n_classes"    : n_classes,
        "features"     : ["Age", "Sex", "BP", "Cholesterol", "Na/K Ratio"],
    })


if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  💊 RxAI Backend — AI Drug Prediction System")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  CV Accuracy   : {cv_acc * 100:.2f}%")
    print(f"  Drug classes  : {n_classes}")
    print(f"  Local Web App : http://localhost:5000")
    print("=" * 52 + "\n")
    app.run(debug=True, port=5000)
