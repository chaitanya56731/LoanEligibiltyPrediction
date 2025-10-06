from flask import Flask, render_template, request, jsonify
import mysql.connector
import pickle

# Load ML model + features
model_data = pickle.load(open("loan_model.pkl", "rb"))
model = model_data["model"]
features = model_data["features"]

# Flask app
app = Flask(__name__)

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",          # change if your MySQL user is different
    password="Shreya$123",  # replace with your MySQL password
    database="loanDB"
)
cursor = db.cursor()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs in correct feature order
        input_data = [float(request.form[feature]) for feature in features]

        # Predict
        prediction = model.predict([input_data])[0]
        result = "Approved" if prediction == 1 else "Rejected"

        # Save into MySQL
        sql = """INSERT INTO loan_applications 
                 (applicant_income, coapplicant_income, loan_amount, tenure, credit_history, prediction) 
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        values = (
            request.form["ApplicantIncome"],
            request.form["CoapplicantIncome"],
            request.form["LoanAmount"],
            request.form["Loan_Amount_Term"],
            request.form["Credit_History"],
            result
        )
        cursor.execute(sql, values)
        db.commit()

        return render_template("result.html", prediction=result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
