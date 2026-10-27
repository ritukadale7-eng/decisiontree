import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Load model if present, otherwise set gracefully
MODEL_PATH = 'decision.pkl'
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model 'decision.pkl' loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Place 'decision.pkl' in the application directory.")

# Embedded UI Template using HTML5 + Tailwind CSS + Lucide Icons + Inter Font
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Placement Predictor | AI Analytics</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Google Font: Inter -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #F8FAFC;
        }
        .gradient-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        }
        .card-shadow {
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
        }
        .custom-ring:focus-within {
            ring-width: 2px;
            ring-color: #3B82F6;
            border-color: transparent;
        }
    </style>
</head>
<body class="min-h-screen flex flex-col text-slate-800 antialiased">

    <!-- App Header -->
    <header class="gradient-header text-white pt-8 pb-16 px-4 sm:px-6 lg:px-8 shadow-lg relative overflow-hidden">
        <div class="absolute inset-0 opacity-10 bg-[radial-gradient(#38bdf8_1px,transparent_1px)] [background-size:16px_16px]"></div>
        <div class="max-w-5xl mx-auto relative z-10 text-center sm:text-left flex flex-col sm:flex-row items-center justify-between gap-4">
            <div>
                <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 text-xs font-medium text-sky-400 mb-3">
                    <i data-lucide="sparkles" class="w-3.5 h-3.5"></i> Decision Tree ML Model
                </div>
                <h1 class="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
                    Job Placement Status Predictor
                </h1>
                <p class="mt-2 text-slate-300 text-sm sm:text-base max-w-xl">
                    Fill in the candidate's academic and background details to predict their campus placement probability.
                </p>
            </div>
            <div class="hidden md:flex items-center gap-3 bg-slate-800/50 backdrop-blur-md p-4 rounded-xl border border-slate-700">
                <div class="p-3 bg-blue-500/20 text-blue-400 rounded-lg">
                    <i data-lucide="brain-circuit" class="w-8 h-8"></i>
                </div>
                <div class="text-left">
                    <div class="text-xs text-slate-400 font-medium">Model Status</div>
                    <div class="text-sm font-semibold {{ 'text-emerald-400' if model_loaded else 'text-amber-400' }}">
                        {{ 'Active & Ready' if model_loaded else 'Model File Missing' }}
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content Container -->
    <main class="flex-grow max-w-5xl w-full mx-auto px-4 sm:px-6 lg:px-8 -mt-8 mb-12 relative z-20">
        <form method="POST" action="/predict" class="space-y-6">
            
            <!-- Section 1: Demographics & Schooling -->
            <div class="bg-white rounded-2xl p-6 sm:p-8 card-shadow border border-slate-200/80 transition-all duration-300 hover:shadow-xl">
                <div class="flex items-center gap-3 pb-4 mb-6 border-b border-slate-100">
                    <div class="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
                        <i data-lucide="graduation-cap" class="w-5 h-5"></i>
                    </div>
                    <div>
                        <h2 class="text-lg font-bold text-slate-900">Secondary Education & Background</h2>
                        <p class="text-xs text-slate-500">Gender and high school performance parameters</p>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Gender -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">Gender</label>
                        <select name="gender" required class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                            <option value="1" {{ 'selected' if form_data.gender == '1' }}>Male</option>
                            <option value="0" {{ 'selected' if form_data.gender == '0' }}>Female</option>
                        </select>
                    </div>

                    <!-- SSC Percentage -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">SSC Score (10th %)</label>
                        <input type="number" step="0.01" min="0" max="100" name="ssc_p" required placeholder="e.g. 67.00" value="{{ form_data.ssc_p or '' }}"
                            class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                    </div>

                    <!-- SSC Board -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">SSC Board</label>
                        <select name="ssc_b" required class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                            <option value="0" {{ 'selected' if form_data.ssc_b == '0' }}>Central</option>
                            <option value="1" {{ 'selected' if form_data.ssc_b == '1' }}>Others / State</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Section 2: Higher Secondary (12th) -->
            <div class="bg-white rounded-2xl p-6 sm:p-8 card-shadow border border-slate-200/80 transition-all duration-300 hover:shadow-xl">
                <div class="flex items-center gap-3 pb-4 mb-6 border-b border-slate-100">
                    <div class="p-2.5 bg-blue-50 text-blue-600 rounded-xl">
                        <i data-lucide="book-open" class="w-5 h-5"></i>
                    </div>
                    <div>
                        <h2 class="text-lg font-bold text-slate-900">Higher Secondary Education (12th)</h2>
                        <p class="text-xs text-slate-500">Board exam metrics and specialized stream choice</p>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- HSC Percentage -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">HSC Score (12th %)</label>
                        <input type="number" step="0.01" min="0" max="100" name="hsc_p" required placeholder="e.g. 70.00" value="{{ form_data.hsc_p or '' }}"
                            class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                    </div>

                    <!-- HSC Board -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">HSC Board</label>
                        <select name="hsc_b" required class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                            <option value="0" {{ 'selected' if form_data.hsc_b == '0' }}>Central</option>
                            <option value="1" {{ 'selected' if form_data.hsc_b == '1' }}>Others / State</option>
                        </select>
                    </div>

                    <!-- HSC Stream -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">HSC Stream</label>
                        <select name="hsc_s" required class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                            <option value="0" {{ 'selected' if form_data.hsc_s == '0' }}>Commerce</option>
                            <option value="1" {{ 'selected' if form_data.hsc_s == '1' }}>Science</option>
                            <option value="2" {{ 'selected' if form_data.hsc_s == '2' }}>Arts</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Section 3: Undergraduation & Experience -->
            <div class="bg-white rounded-2xl p-6 sm:p-8 card-shadow border border-slate-200/80 transition-all duration-300 hover:shadow-xl">
                <div class="flex items-center gap-3 pb-4 mb-6 border-b border-slate-100">
                    <div class="p-2.5 bg-emerald-50 text-emerald-600 rounded-xl">
                        <i data-lucide="award" class="w-5 h-5"></i>
                    </div>
                    <div>
                        <h2 class="text-lg font-bold text-slate-900">Undergraduate Academic Record</h2>
                        <p class="text-xs text-slate-500">Degree details, percentage score, and work history</p>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Degree Percentage -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">Degree Score (%)</label>
                        <input type="number" step="0.01" min="0" max="100" name="degree_p" required placeholder="e.g. 65.00" value="{{ form_data.degree_p or '' }}"
                            class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                    </div>

                    <!-- Degree Type -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">Degree Field</label>
                        <select name="degree_t" required class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                            <option value="1" {{ 'selected' if form_data.degree_t == '1' }}>Comm & Mgmt</option>
                            <option value="0" {{ 'selected' if form_data.degree_t == '0' }}>Sci & Tech</option>
                            <option value="2" {{ 'selected' if form_data.degree_t == '2' }}>Others</option>
                        </select>
                    </div>

                    <!-- Work Experience -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">Work Experience</label>
                        <select name="workex" required class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                            <option value="0" {{ 'selected' if form_data.workex == '0' }}>No Experience</option>
                            <option value="1" {{ 'selected' if form_data.workex == '1' }}>Has Prior Experience</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Section 4: Testing & MBA Specialization -->
            <div class="bg-white rounded-2xl p-6 sm:p-8 card-shadow border border-slate-200/80 transition-all duration-300 hover:shadow-xl">
                <div class="flex items-center gap-3 pb-4 mb-6 border-b border-slate-100">
                    <div class="p-2.5 bg-amber-50 text-amber-600 rounded-xl">
                        <i data-lucide="briefcase" class="w-5 h-5"></i>
                    </div>
                    <div>
                        <h2 class="text-lg font-bold text-slate-900">Post-Graduation & Aptitude</h2>
                        <p class="text-xs text-slate-500">Employability test scores and MBA specialization metrics</p>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- E-Test Percentage -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">E-Test Score (%)</label>
                        <input type="number" step="0.01" min="0" max="100" name="etest_p" required placeholder="e.g. 55.00" value="{{ form_data.etest_p or '' }}"
                            class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                    </div>

                    <!-- Specialization -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">MBA Specialization</label>
                        <select name="specialisation" required class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                            <option value="0" {{ 'selected' if form_data.specialisation == '0' }}>Mkt & Fin</option>
                            <option value="1" {{ 'selected' if form_data.specialisation == '1' }}>Mkt & HR</option>
                        </select>
                    </div>

                    <!-- MBA Percentage -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">MBA Score (%)</label>
                        <input type="number" step="0.01" min="0" max="100" name="mba_p" required placeholder="e.g. 58.80" value="{{ form_data.mba_p or '' }}"
                            class="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-900 transition-all">
                    </div>
                </div>
            </div>

            <!-- Form Controls -->
            <div class="flex flex-col sm:flex-row items-center justify-end gap-4 pt-2">
                <a href="/" class="w-full sm:w-auto px-6 py-3 rounded-xl border border-slate-300 text-slate-700 font-medium text-center text-sm hover:bg-slate-100 transition-all">
                    Reset Form
                </a>
                <button type="submit" class="w-full sm:w-auto px-8 py-3 bg-slate-900 hover:bg-slate-800 active:scale-98 text-white font-semibold text-sm rounded-xl shadow-lg shadow-slate-900/20 transition-all flex items-center justify-center gap-2">
                    <i data-lucide="cpu" class="w-4 h-4"></i> Run Prediction
                </button>
            </div>
        </form>

        <!-- Prediction Result Display -->
        {% if result %}
        <div id="prediction-result" class="mt-8 scroll-mt-6">
            {% if result.status == 'Placed' %}
            <div class="p-6 sm:p-8 rounded-2xl bg-emerald-50 border border-emerald-200 card-shadow flex flex-col sm:flex-row items-center justify-between gap-6">
                <div class="flex items-center gap-5">
                    <div class="w-16 h-16 rounded-2xl bg-emerald-500 text-white flex items-center justify-center shrink-0 shadow-lg shadow-emerald-500/30">
                        <i data-lucide="check-circle-2" class="w-10 h-10"></i>
                    </div>
                    <div>
                        <span class="inline-block px-3 py-1 rounded-full bg-emerald-100 text-emerald-800 text-xs font-bold uppercase tracking-wider mb-1">High Probability</span>
                        <h3 class="text-2xl font-bold text-emerald-950">Candidate Placed</h3>
                        <p class="text-xs sm:text-sm text-emerald-700 mt-1">Based on the provided academic and professional profile, the candidate is likely to secure a placement.</p>
                    </div>
                </div>
            </div>
            {% elif result.status == 'Not Placed' %}
            <div class="p-6 sm:p-8 rounded-2xl bg-rose-50 border border-rose-200 card-shadow flex flex-col sm:flex-row items-center justify-between gap-6">
                <div class="flex items-center gap-5">
                    <div class="w-16 h-16 rounded-2xl bg-rose-500 text-white flex items-center justify-center shrink-0 shadow-lg shadow-rose-500/30">
                        <i data-lucide="x-circle" class="w-10 h-10"></i>
                    </div>
                    <div>
                        <span class="inline-block px-3 py-1 rounded-full bg-rose-100 text-rose-800 text-xs font-bold uppercase tracking-wider mb-1">Needs Improvement</span>
                        <h3 class="text-2xl font-bold text-rose-950">Not Placed</h3>
                        <p class="text-xs sm:text-sm text-rose-700 mt-1">The current parameters indicate a lower probability of campus placement. Focus on enhancing test scores or gaining work experience.</p>
                    </div>
                </div>
            </div>
            {% else %}
            <div class="p-6 rounded-2xl bg-amber-50 border border-amber-200 text-amber-800 text-sm flex items-center gap-3">
                <i data-lucide="alert-triangle" class="w-5 h-5 shrink-0"></i>
                <div>{{ result.error }}</div>
            </div>
            {% endif %}
        </div>
        {% endif %}
    </main>

    <!-- Footer -->
    <footer class="bg-white border-t border-slate-200 mt-auto py-6 text-center text-xs text-slate-500">
        <div class="max-w-5xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
            <div>Campus Placement Intelligence Platform &bull; Powered by DecisionTree Classifier</div>
            <div>Scikit-Learn & Flask</div>
        </div>
    </footer>

    <!-- Initialize Lucide Icons & Auto Scroll -->
    <script>
        lucide.createIcons();
        {% if result %}
        document.getElementById('prediction-result')?.scrollIntoView({ behavior: 'smooth' });
        {% endif %}
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    """Render home interface with blank inputs"""
    return render_template_string(
        HTML_LAYOUT, 
        model_loaded=(model is not None),
        form_data={},
        result=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handle model inference from form submit"""
    form_data = request.form.to_dict()
    result = None

    if not model:
        result = {'error': 'The decision.pkl model file was not found on the server. Please verify setup.'}
        return render_template_string(
            HTML_LAYOUT, 
            model_loaded=False, 
            form_data=form_data, 
            result=result
        )

    try:
        # Extract features ordered strictly as specified by decision.pkl schema:
        # [gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p]
        features = [
            float(form_data.get('gender', 0)),
            float(form_data.get('ssc_p', 0.0)),
            float(form_data.get('ssc_b', 0)),
            float(form_data.get('hsc_p', 0.0)),
            float(form_data.get('hsc_b', 0)),
            float(form_data.get('hsc_s', 0)),
            float(form_data.get('degree_p', 0.0)),
            float(form_data.get('degree_t', 0)),
            float(form_data.get('workex', 0)),
            float(form_data.get('etest_p', 0.0)),
            float(form_data.get('specialisation', 0)),
            float(form_data.get('mba_p', 0.0))
        ]

        # Execute decision tree prediction
        prediction = model.predict(np.array([features]))[0]

        # Class mapping logic for string or numeric target output
        classes = ['Not Placed', 'Placed']
        if isinstance(prediction, (np.integer, int)):
            predicted_status = classes[prediction] if prediction < len(classes) else str(prediction)
        else:
            predicted_status = str(prediction)

        result = {'status': predicted_status}

    except Exception as e:
        result = {'error': f'Prediction Failed: {str(e)}'}

    return render_template_string(
        HTML_LAYOUT, 
        model_loaded=True, 
        form_data=form_data, 
        result=result
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for programmatic inferences"""
    if not model:
        return jsonify({'error': 'Model decision.pkl is not loaded.'}), 500

    try:
        data = request.get_json(force=True)
        features = [
            float(data['gender']),
            float(data['ssc_p']),
            float(data['ssc_b']),
            float(data['hsc_p']),
            float(data['hsc_b']),
            float(data['hsc_s']),
            float(data['degree_p']),
            float(data['degree_t']),
            float(data['workex']),
            float(data['etest_p']),
            float(data['specialisation']),
            float(data['mba_p'])
        ]
        
        pred = model.predict([features])[0]
        classes = ['Not Placed', 'Placed']
        status = classes[int(pred)] if isinstance(pred, (np.integer, int)) and int(pred) < len(classes) else str(pred)
        
        return jsonify({'prediction': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
