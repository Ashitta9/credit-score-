from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the machine learning pipeline
pipeline = joblib.load("pipeline.sav")

app = Flask(__name__)

# Define a function to create and fit label encoders
def create_label_encoders(data):
    label_encoders = {}
    categorical_features = ["Occupation", "Credit_Mix", "Payment_Behaviour"]
    for feature in categorical_features:
        encoder = LabelEncoder()
        encoder.fit(data[feature])
        label_encoders[feature] = encoder
    return label_encoders

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Retrieve form data
        Month = float(request.form["Month"])
        Age = float(request.form["Age"])
        Occupation = request.form["Occupation"]
        Annual_Income = float(request.form["Annual_Income"])
        Monthly_Inhand_Salary = float(request.form["Monthly_Inhand_Salary"])
        Num_Bank_Accounts = int(request.form["Num_Bank_Accounts"])
        Num_Credit_Card = int(request.form["Num_Credit_Card"])
        Interest_Rate = int(request.form["Interest_Rate"])
        Num_of_Loan = float(request.form["Num_of_Loan"])
        Delay_from_due_date = int(request.form["Delay_from_due_date"])
        Num_of_Delayed_Payment = float(request.form["Num_of_Delayed_Payment"])
        Changed_Credit_Limit = float(request.form["Changed_Credit_Limit"])
        Num_Credit_Inquiries = float(request.form["Num_Credit_Inquiries"])
        Credit_Mix = request.form["Credit_Mix"]
        Outstanding_Debt = float(request.form["Outstanding_Debt"])
        Credit_Utilization_Ratio = float(request.form["Credit_Utilization_Ratio"])
        Credit_History_Age = int(request.form["Credit_History_Age"])
        Payment_of_Min_Amount = float(request.form["Payment_of_Min_Amount"])
        Total_EMI_per_month = float(request.form["Total_EMI_per_month"])
        Amount_invested_monthly = float(request.form["Amount_invested_monthly"])
        Payment_Behaviour = request.form["Payment_Behaviour"]
        Monthly_Balance = float(request.form["Monthly_Balance"])

        # Create DataFrame from form data
        df = pd.DataFrame({
            "Month": [Month],
            "Age": [Age],
            "Occupation": [Occupation],
            "Annual_Income": [Annual_Income],
            "Monthly_Inhand_Salary": [Monthly_Inhand_Salary],
            "Num_Bank_Accounts": [Num_Bank_Accounts],
            "Num_Credit_Card": [Num_Credit_Card],
            "Interest_Rate": [Interest_Rate],
            "Num_of_Loan": [Num_of_Loan],
            "Delay_from_due_date": [Delay_from_due_date],
            "Num_of_Delayed_Payment": [Num_of_Delayed_Payment],
            "Changed_Credit_Limit": [Changed_Credit_Limit],
            "Num_Credit_Inquiries": [Num_Credit_Inquiries],
            "Credit_Mix": [Credit_Mix],
            "Outstanding_Debt": [Outstanding_Debt],
            "Credit_Utilization_Ratio": [Credit_Utilization_Ratio],
            "Credit_History_Age": [Credit_History_Age],
            "Payment_of_Min_Amount": [Payment_of_Min_Amount],
            "Total_EMI_per_month": [Total_EMI_per_month],
            "Amount_invested_monthly": [Amount_invested_monthly],
            "Payment_Behaviour": [Payment_Behaviour],
            "Monthly_Balance": [Monthly_Balance]
        })

        # Create label encoders if not available
        if 'label_encoders' not in globals():
            label_encoders = create_label_encoders(df)

        # Encode categorical features
        df_encoded = df.copy()
        for feature in label_encoders:
            df_encoded[feature] = label_encoders[feature].transform(df[feature])

        # Predict credit score using the pipeline
        credit_score = pipeline.predict(df_encoded)[0]

        # Render score page with prediction
        return redirect(url_for('score', credit_score=credit_score))

    # Render form on GET request
    return render_template("credit.html")

@app.route("/score/<int:credit_score>")
def score(credit_score):
    # Render score page with prediction
    return render_template("score.html", credit_score=credit_score)

if __name__ == "__main__":
    app.run(debug=True)
