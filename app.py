from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('random_forest_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get feature values from form
        features = [float(x) for x in request.form.values()]
        final_input = np.array([features])
        
        # Make prediction
        prediction = model.predict(final_input)[0]
        
        return render_template('index.html', prediction_text=f'Predicted Class: {prediction}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
