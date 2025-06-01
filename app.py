from flask import Flask, request
import numpy as np
import pickle

# Load the model
model = pickle.load(open("random_forest_classifier.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# HTML form embedded inside the Python file
html_form = """
<!doctype html>
<html>
<head>
    <title>ML Prediction App</title>
</head>
<body style="font-family: Arial; margin: 40px;">
    <h2>Enter Input Features</h2>
    <form action="/predict" method="post">
        <label>Feature 1:</label><br>
        <input type="text" name="feature1"><br><br>
        <label>Feature 2:</label><br>
        <input type="text" name="feature2"><br><br>
        <label>Feature 3:</label><br>
        <input type="text" name="feature3"><br><br>
        <label>Feature 4:</label><br>
        <input type="text" name="feature4"><br><br>
        <!-- Add more fields if your model has more features -->
        <input type="submit" value="Predict">
    </form>
    <br>
    <h3>{}</h3>
</body>
</html>
"""

# Home route
@app.route("/", methods=["GET"])
def index():
    return html_form.format("")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect features from the form
        features = [
            float(request.form["feature1"]),
            float(request.form["feature2"]),
            float(request.form["feature3"]),
            float(request.form["feature4"])
            # Add more if needed
        ]
        final_input = np.array(features).reshape(1, -1)
        prediction = model.predict(final_input)[0]

        result = f"Predicted output: {prediction}"
        return html_form.format(result)
    
    except Exception as e:
        return html_form.format(f"Error: {str(e)}")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
