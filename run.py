"""
RxAI Drug Prediction System Runner
Enables launching RxAI backend directly.
Usage:
    python run.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app, test_acc, cv_acc, n_classes

if __name__ == "__main__":
    print("\n" + "=" * 48)
    print("  RxAI Backend — Drug Prediction System")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  CV Accuracy   : {cv_acc * 100:.2f}%")
    print(f"  Drug classes  : {n_classes}")
    print(f"  Web Interface : http://localhost:5000")
    print("=" * 48 + "\n")
    app.run(debug=True, port=5000)
