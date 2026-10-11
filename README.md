# 💊 RxAI - Drug Prediction System

A Machine Learning web application that predicts the most suitable drug based on patient information such as age, gender, blood pressure, cholesterol level, and sodium-to-potassium ratio.

## 🚀 Features

* Predicts suitable drugs using Machine Learning
* Simple and user-friendly interface
* Fast prediction results
* Flask-based web application
* Real-world healthcare dataset

## 🛠️ Technologies Used

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* HTML
* CSS
* JavaScript

## 📊 Input Parameters

The model uses:

* Age
* Gender
* Blood Pressure
* Cholesterol Level
* Na/K Ratio

## ⚙️ How to Run

### Clone Repository

```bash
git clone https://github.com/Vamshiyadavk/RxAI-Drug-Prediction-System.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

### Open Browser

```text
http://127.0.0.1:5000
```

## 📂 Project Structure

```text
RxAI/
│
├── app.py                     # Flask web server & prediction API
├── run.py                     # Convenience runner script
├── train_model.py             # Script to train and export the ML model
├── drug_dataset.csv           # Clinical dataset for model training
├── drug_descriptions.json     # Drug classes, indications & details
├── drug_model.pkl             # Pre-trained Random Forest model bundle
├── drug_prediction_model.ipynb# Jupyter notebook for EDA & modeling
├── index.html                 # Modern frontend UI
└── requirements.txt           # Python dependencies
```

## 🎯 Project Objective

The goal of this project is to use Machine Learning techniques to predict suitable drugs based on patient information and demonstrate the practical application of AI in healthcare.

## 📚 What I Learned

* Machine Learning model building
* Data preprocessing
* Flask web development
* Frontend and backend integration
* Healthcare data analysis

## ⚠️ Disclaimer

This project is created for educational purposes only and should not be used as a substitute for professional medical advice.

## 👨‍💻 Author

Vamshi Yadav Kodi

B.Tech Data Science Student

GitHub: https://github.com/Vamshiyadavk
