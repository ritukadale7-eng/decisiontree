import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Decision Tree model
MODEL_PATH = 'decision.pkl'
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# HTML Layout with embedded CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Campus Placement Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #10b981;
            --danger: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .container {
            background: var(--card-bg);
            width: 100%;
            max-width: 800px;
            padding: 2.5rem;
            border-radius: 1rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 1.875rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 0.375rem;
        }

        .form-group input, .form-group select {
            padding: 0.625rem 0.875rem;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
            background-color: #fff;
        }

        .form-group input:focus, .form-group select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 0.875rem;
            background-color: var(--primary);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }

        .btn-submit:hover {
            background-color: var(--primary-hover);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.25rem;
            border-radius: 0.5rem;
            text-align: center;
            font-size: 1.25rem;
            font-weight: 700;
        }

        .result-placed {
            background-color: #ecfdf5;
            color: var(--success);
            border: 1px solid #a7f3d0;
        }

        .result-not-placed {
            background-color: #fef2f2;
            color: var(--danger);
            border: 1px solid #fecaca;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Campus Placement Prediction</h1>
        <p>Enter student details to evaluate placement eligibility</p>
    </div>

    <form action="/predict" method="POST">
        <div class="form-grid">
            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" required>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>

            <div class="form-group">
                <label for="ssc_p">10th Percentage (ssc_p)</label>
                <input type="number" step="0.01" id="ssc_p" name="ssc_p" placeholder="e.g. 67.0" required>
            </div>

            <div class="form-group">
                <label for="ssc_b">10th Board (ssc_b)</label>
                <select id="ssc_b" name="ssc_b" required>
                    <option value="1">Central</option>
                    <option value="0">Others</option>
                </select>
            </div>

            <div class="form-group">
                <label for="hsc_p">12th Percentage (hsc_p)</label>
                <input type="number" step="0.01" id="hsc_p" name="hsc_p" placeholder="e.g. 70.0" required>
            </div>

            <div class="form-group">
                <label for="hsc_b">12th Board (hsc_b)</label>
                <select id="hsc_b" name="hsc_b" required>
                    <option value="1">Central</option>
                    <option value="0">Others</option>
                </select>
            </div>

            <div class="form-group">
                <label for="hsc_s">12th Specialization (hsc_s)</label>
                <select id="hsc_s" name="hsc_s" required>
                    <option value="0">Commerce</option>
                    <option value="1">Science</option>
                    <option value="2">Arts</option>
                </select>
            </div>

            <div class="form-group">
                <label for="degree_p">Degree Percentage (degree_p)</label>
                <input type="number" step="0.01" id="degree_p" name="degree_p" placeholder="e.g. 65.0" required>
            </div>

            <div class="form-group">
                <label for="degree_t">Degree Type (degree_t)</label>
                <select id="degree_t" name="degree_t" required>
                    <option value="0">Sci&Tech</option>
                    <option value="1">Comm&Mgmt</option>
                    <option value="2">Others</option>
                </select>
            </div>

            <div class="form-group">
                <label for="workex">Work Experience (workex)</label>
                <select id="workex" name="workex" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="etest_p">E-Test Percentage (etest_p)</label>
                <input type="number" step="0.01" id="etest_p" name="etest_p" placeholder="e.g. 55.0" required>
            </div>

            <div class="form-group">
                <label for="specialisation">MBA Specialization</label>
                <select id="specialisation" name="specialisation" required>
                    <option value="1">Mkt&HR</option>
                    <option value="0">Mkt&Fin</option>
                </select>
            </div>

            <div class="form-group">
                <label for="mba_p">MBA Percentage (mba_p)</label>
                <input type="number" step="0.01" id="mba_p" name="mba_p" placeholder="e.g. 58.8" required>
            </div>

            <button type="submit" class="btn-submit">Predict Status</button>
        </div>
    </form>

    {% if prediction %}
        <div class="result-box {% if prediction == 'Placed' %}result-placed{% else %}result-not-placed{% endif %}">
            Prediction Outcome: {{ prediction }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model file 'decision.pkl' not found.", 500

    try:
        # Extract features in the exact expected order
        feature_values = [
            float(request.form['gender']),
            float(request.form['ssc_p']),
            float(request.form['ssc_b']),
            float(request.form['hsc_p']),
            float(request.form['hsc_b']),
            float(request.form['hsc_s']),
            float(request.form['degree_p']),
            float(request.form['degree_t']),
            float(request.form['workex']),
            float(request.form['etest_p']),
            float(request.form['specialisation']),
            float(request.form['mba_p'])
        ]

        # Reshape input for single-sample prediction
        input_data = np.array(feature_values).reshape(1, -1)
        
        # Predict class
        prediction_result = model.predict(input_data)[0]

        return render_template_string(HTML_TEMPLATE, prediction=prediction_result)

    except Exception as e:
        return f"Error making prediction: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)