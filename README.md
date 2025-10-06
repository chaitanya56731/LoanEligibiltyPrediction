# 🏦 Loan Eligibility Prediction

A machine learning + Flask web application to predict loan eligibility based on applicant details.  
Built using **Python, Flask, Scikit-learn, MySQL, and HTML/CSS**.  

---

## 🚀 Features
- Predicts whether a loan application is **Approved** or **Rejected**.  
- Uses **Random Forest Classifier** trained on loan dataset.  
- User-friendly **web form** built with Flask & HTML.  
- Stores application details and prediction results in **MySQL database**.  

---

## 📂 Project Structure
LoanEligibilityPrediction/
│── app.py # Flask backend
│── train_model.py # ML model training script
│── loan_model.pkl # Saved trained model
│── loan.csv # Dataset
│── requirements.txt # Dependencies
│── README.md # Project documentation
│
├── templates/
│ └── index.html # Frontend form
│
├── static/
│ └── style.css # CSS styles
