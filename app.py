from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        int(request.form["pclass"]),
        int(request.form["sex"]),
        float(request.form["age"]),
        int(request.form["sibsp"]),
        int(request.form["parch"]),
        float(request.form["fare"]),
        int(request.form["embarked"])
    ]

    final_features = np.array([features])
    prediction = model.predict(final_features)[0]

    result = "Survived 🟢" if prediction == 1 else "Not Survived 🔴"
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True,port=9000)



