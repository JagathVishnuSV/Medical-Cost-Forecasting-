from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load(r"C:\Users\jagat\Downloads\Medical cost forecasting ds project\model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve form data
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = int(request.form["smoker"])
        region_northeast = int(request.form["region_northeast"])
        region_northwest = int(request.form["region_northwest"])
        region_southeast = int(request.form["region_southeast"])
        region_southwest = int(request.form["region_southwest"])
        sex = int(request.form["sex"])

        # Arrange data into the correct format for prediction
        input_data = np.array([[age, bmi, children, smoker, region_northeast, 
                                region_northwest, region_southeast, region_southwest, sex]])

        # Make a prediction
        predicted_charge = model.predict(input_data)

        # Return the prediction result
        return render_template("result.html", prediction=predicted_charge[0])

    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True)
