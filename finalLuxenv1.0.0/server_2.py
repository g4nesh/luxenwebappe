import os
import io
import json
import base64
import requests
from datetime import datetime

import boto3
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG (can remain outside or be part of a larger config class) ---
S3_BUCKET = os.environ.get('S3_BUCKET_NAME', 'luxen-test-storage-v1').strip()
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') # Use .get for safer access if not set

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

GEMINI_API_URL = (
     "https://generativelanguage.googleapis.com/"
     "v1beta/models/gemini-2.0-flash:generateContent"
     f"?key={GEMINI_API_KEY}"
)

print(f"Initialized with GEMINI_API_KEY: {GEMINI_API_KEY[:5]}...")  # Print first 5 chars for verification

METRICS = [
    "Redness Level",
    "Scaling Level",
    "Texture Score",
    "Color Variation",
    "Severity Score",
    "Predicted Deficiency"
] 

class EczemaAnalyzer:
    def __init__(self, s3_bucket: str, gemini_api_key: str, gemini_api_url: str):
        self.s3_bucket = s3_bucket
        self.gemini_api_key = gemini_api_key
        self.gemini_api_url = gemini_api_url
        self.s3_client = boto3.client('s3')

    # --- Gemini Methods ---
    def _generate_content_with_gemini(self, prompt: str, base64_image: str) -> dict:
        """
        Internal helper to interact with the Gemini API.
        """
        print(f"Inside _generate_content_with_gemini. self.gemini_api_key: {self.gemini_api_key}")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64_image
                        }
                    }
                ]
            }]
        }
        print(f"Sending prompt to Gemini: {prompt[:200]}...") # Log the start of the prompt
        r = requests.post(self.gemini_api_url, headers=headers, json=body)
        print(f"Gemini API response status code: {r.status_code}") # Log the status code
        print(f"Gemini API raw response text: {r.text[:500]}...") # Log the raw response text
        r.raise_for_status()
        cand = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        # strip ```json fences
        txt = cand.strip()
        if txt.startswith("```json"):
            txt = txt[len("```json"):].strip()
        if txt.endswith("```"):
            txt = txt[:-3].strip()
        print(f"Parsed Gemini response text: {txt[:500]}...") # Log the parsed text
        return json.loads(txt)

    def get_gemini_analysis_results(self, image_bytes: bytes) -> dict:
        """
        Analyzes an image using Gemini and returns the eczema metrics.
        This is a method to get Gemini results.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            "Analyze this image of a hand with eczema and provide the following metrics "
            "as integer percentages from 1 to 100 (where 100 represents the worst severity): "
            "Redness Level, Scaling Level, Texture Score, Color Variation, Severity Score. "
            "Return only a valid JSON object with keys matching these metric names exactly "
            "and values as integers from 1 to 100."
        )
        result = self._generate_content_with_gemini(prompt, b64)

        # Clean and clamp output to integers 1–100
        cleaned = {}
        for k, v in result.items():
            try:
                val = int(float(v))
            except (ValueError, TypeError):
                # Robustly extract digits if conversion fails
                digits = ''.join(filter(str.isdigit, str(v)))
                val = int(digits) if digits else 0
            cleaned[k] = max(1, min(100, val))
        
        # Add predicted deficiency based on metrics
        cleaned['Predicted Deficiency'] = self._determine_deficiency(cleaned)
        return cleaned

    def _determine_deficiency(self, metrics: dict) -> list:
        """
        Determines potential deficiencies based on relative metric patterns.
        Returns a list of top 3 most likely deficiencies based on medical research and symptom patterns.
        """
        redness = metrics.get('Redness Level', 0)
        scaling = metrics.get('Scaling Level', 0)
        texture = metrics.get('Texture Score', 0)
        color_var = metrics.get('Color Variation', 0)
        severity = metrics.get('Severity Score', 0)

        # Calculate relative differences
        is_high_scaling = scaling > (redness + texture) / 2
        is_high_redness = redness > (scaling + texture) / 2
        is_high_texture = texture > (redness + scaling) / 2
        is_high_color_var = color_var > (redness + scaling + texture) / 3
        is_severe = severity > 70
        is_moderate = severity > 50
        is_mild = severity > 30

        # Initialize deficiency scores
        deficiency_scores = {
            "Vitamin D": 0,
            "Vitamin B12": 0,
            "Zinc": 0,
            "Essential Fatty Acids": 0,
            "Iron": 0,
            "Vitamin A": 0,
            "Vitamin E": 0,
            "Vitamin C": 0,
            "Biotin": 0,
            "Selenium": 0,
            "Vitamin B6": 0,
            "Vitamin B3 (Niacin)": 0,
            "Copper": 0,
            "Magnesium": 0,
            "Vitamin K": 0
        }

        # Vitamin D deficiency patterns
        if is_high_scaling and is_high_redness:
            deficiency_scores["Vitamin D"] += 3
        if is_high_texture and is_high_scaling:
            deficiency_scores["Vitamin D"] += 2
        if is_severe:
            deficiency_scores["Vitamin D"] += 1

        # Vitamin B12 deficiency patterns
        if is_high_color_var and is_high_texture:
            deficiency_scores["Vitamin B12"] += 3
        if is_high_redness and is_high_texture:
            deficiency_scores["Vitamin B12"] += 2
        if is_moderate:
            deficiency_scores["Vitamin B12"] += 1

        # Zinc deficiency patterns
        if is_high_scaling and is_severe:
            deficiency_scores["Zinc"] += 3
        if is_high_texture and is_high_redness:
            deficiency_scores["Zinc"] += 2
        if is_high_color_var:
            deficiency_scores["Zinc"] += 1

        # Essential fatty acids deficiency patterns
        if is_high_scaling and is_high_texture and not is_high_redness:
            deficiency_scores["Essential Fatty Acids"] += 3
        if is_high_texture and is_mild:
            deficiency_scores["Essential Fatty Acids"] += 2
        if is_high_color_var and not is_high_redness:
            deficiency_scores["Essential Fatty Acids"] += 1

        # Iron deficiency patterns
        if is_high_color_var and is_high_redness and not is_high_scaling:
            deficiency_scores["Iron"] += 3
        if is_high_redness and is_moderate:
            deficiency_scores["Iron"] += 2
        if is_high_texture and not is_high_scaling:
            deficiency_scores["Iron"] += 1

        # Vitamin A deficiency patterns
        if is_high_texture and is_high_scaling and not is_high_redness:
            deficiency_scores["Vitamin A"] += 3
        if is_high_scaling and is_moderate:
            deficiency_scores["Vitamin A"] += 2
        if is_high_texture and not is_high_color_var:
            deficiency_scores["Vitamin A"] += 1

        # Vitamin E deficiency patterns
        if is_high_texture and is_high_scaling and not is_high_color_var:
            deficiency_scores["Vitamin E"] += 3
        if is_high_scaling and is_mild:
            deficiency_scores["Vitamin E"] += 2
        if is_high_texture and not is_high_redness:
            deficiency_scores["Vitamin E"] += 1

        # Vitamin C deficiency patterns
        if is_high_texture and is_high_redness and not is_high_scaling:
            deficiency_scores["Vitamin C"] += 3
        if is_high_redness and is_mild:
            deficiency_scores["Vitamin C"] += 2
        if is_high_texture and not is_high_color_var:
            deficiency_scores["Vitamin C"] += 1

        # Biotin deficiency patterns
        if is_high_scaling and is_high_redness and is_high_texture:
            deficiency_scores["Biotin"] += 3
        if is_high_scaling and is_moderate:
            deficiency_scores["Biotin"] += 2
        if is_high_texture and is_high_color_var:
            deficiency_scores["Biotin"] += 1

        # Selenium deficiency patterns
        if is_high_texture and is_high_scaling and is_high_color_var:
            deficiency_scores["Selenium"] += 3
        if is_high_texture and is_severe:
            deficiency_scores["Selenium"] += 2
        if is_high_color_var and is_high_scaling:
            deficiency_scores["Selenium"] += 1

        # Vitamin B6 deficiency patterns
        if is_high_redness and is_high_texture and is_high_color_var:
            deficiency_scores["Vitamin B6"] += 3
        if is_high_redness and is_moderate:
            deficiency_scores["Vitamin B6"] += 2
        if is_high_texture and is_high_scaling:
            deficiency_scores["Vitamin B6"] += 1

        # Vitamin B3 (Niacin) deficiency patterns
        if is_high_redness and is_high_scaling and is_high_texture:
            deficiency_scores["Vitamin B3 (Niacin)"] += 3
        if is_high_redness and is_severe:
            deficiency_scores["Vitamin B3 (Niacin)"] += 2
        if is_high_scaling and is_high_color_var:
            deficiency_scores["Vitamin B3 (Niacin)"] += 1

        # Copper deficiency patterns
        if is_high_color_var and is_high_texture and is_high_scaling:
            deficiency_scores["Copper"] += 3
        if is_high_color_var and is_moderate:
            deficiency_scores["Copper"] += 2
        if is_high_texture and is_high_redness:
            deficiency_scores["Copper"] += 1

        # Magnesium deficiency patterns
        if is_high_texture and is_high_redness and is_high_scaling:
            deficiency_scores["Magnesium"] += 3
        if is_high_texture and is_severe:
            deficiency_scores["Magnesium"] += 2
        if is_high_redness and is_high_color_var:
            deficiency_scores["Magnesium"] += 1

        # Vitamin K deficiency patterns
        if is_high_redness and is_high_scaling and is_high_color_var:
            deficiency_scores["Vitamin K"] += 3
        if is_high_redness and is_mild:
            deficiency_scores["Vitamin K"] += 2
        if is_high_scaling and is_high_texture:
            deficiency_scores["Vitamin K"] += 1

        # Sort deficiencies by score and get top 3
        sorted_deficiencies = sorted(deficiency_scores.items(), key=lambda x: x[1], reverse=True)
        top_deficiencies = sorted_deficiencies[:3]

        # Format the results
        results = []
        for deficiency, score in top_deficiencies:
            if score > 0:  # Only include deficiencies with a score
                results.append({
                    "deficiency": deficiency,
                    "score": score,
                    "confidence": "High" if score >= 3 else "Medium" if score >= 2 else "Low",
                    "recommendation": self._get_deficiency_recommendation(deficiency)
                })

        return results if results else [{
            "deficiency": "No specific deficiency pattern detected",
            "score": 0,
            "confidence": "Low",
            "recommendation": "For proper assessment of skin conditions and potential deficiencies, please consult a healthcare provider."
        }]

    def _get_deficiency_recommendation(self, deficiency: str) -> str:
        """Returns specific recommendations for each deficiency type."""
        recommendations = {
            "Vitamin D": "Consider increasing sun exposure (with protection) and consuming vitamin D-rich foods like fatty fish, egg yolks, and fortified dairy products. Consult a healthcare provider for proper supplementation.",
            "Vitamin B12": "Include more animal products, fortified foods, or consider B12 supplements. A healthcare provider can perform proper testing and recommend appropriate treatment.",
            "Zinc": "Increase intake of zinc-rich foods like oysters, red meat, poultry, beans, and nuts. Professional medical assessment is recommended for proper diagnosis.",
            "Essential Fatty Acids": "Increase consumption of omega-3 rich foods like fatty fish, flaxseeds, and walnuts. Consider consulting a healthcare provider for proper evaluation.",
            "Iron": "Include more iron-rich foods like red meat, spinach, and legumes. Blood tests are recommended for proper diagnosis.",
            "Vitamin A": "Increase intake of vitamin A-rich foods like sweet potatoes, carrots, and leafy greens. Professional medical assessment is advised.",
            "Vitamin E": "Include more vitamin E-rich foods like nuts, seeds, and vegetable oils. Consult a healthcare provider for proper evaluation.",
            "Vitamin C": "Increase consumption of citrus fruits, berries, and leafy greens. Professional medical assessment is recommended.",
            "Biotin": "Include more biotin-rich foods like eggs, nuts, and whole grains. A healthcare provider can perform proper testing.",
            "Selenium": "Increase intake of selenium-rich foods like Brazil nuts, seafood, and whole grains. Professional medical assessment is advised.",
            "Vitamin B6": "Include more B6-rich foods like fish, poultry, and bananas. Consult a healthcare provider for proper evaluation.",
            "Vitamin B3 (Niacin)": "Increase consumption of niacin-rich foods like meat, fish, and whole grains. Professional medical assessment is recommended.",
            "Copper": "Include more copper-rich foods like shellfish, nuts, and seeds. Consult a healthcare provider for proper evaluation.",
            "Magnesium": "Increase intake of magnesium-rich foods like leafy greens, nuts, and whole grains. Professional medical assessment is advised.",
            "Vitamin K": "Include more vitamin K-rich foods like leafy greens, broccoli, and fermented foods. Consult a healthcare provider for proper evaluation."
        }
        return recommendations.get(deficiency, "Consult a healthcare provider for proper assessment and treatment recommendations.")

    # --- S3 Methods ---
    def _get_s3_key(self, user_id: str, metric: str) -> str:
        """Helper to generate S3 object key."""
        return f"{user_id}/{metric.replace(' ', '_')}.csv"

    def get_metrics_from_s3(self, user_id: str, metric: str) -> pd.DataFrame:
        """
        Retrieves a user's metric data from S3.
        This is a method to call from AWS (S3).
        """
        key = self._get_s3_key(user_id, metric)
        try:
            obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
            return pd.read_csv(io.BytesIO(obj['Body'].read()))
        except self.s3_client.exceptions.NoSuchKey:
            return pd.DataFrame(columns=["timestamp", "value"])
        except Exception as e:
            print(f"Error downloading metric {metric} for {user_id} from S3: {e}")
            return pd.DataFrame(columns=["timestamp", "value"])


    def send_metrics_to_s3(self, user_id: str, metrics_data: dict):
        """
        Saves multiple metrics for a user to S3.
        This is a method to send to AWS (S3).
        """
        print(f"Attempting to send metrics to S3 for user: {user_id}")
        print(f"Metrics data received: {metrics_data}")
        ts = datetime.utcnow().isoformat()
        for metric, val in metrics_data.items():
            print(f"Processing metric: {metric} with value: {val}")
            df = self.get_metrics_from_s3(user_id, metric)
            # Use pd.concat for appending for newer pandas versions
            new_row_df = pd.DataFrame([{"timestamp": ts, "value": val}])
            df = pd.concat([df, new_row_df], ignore_index=True)

            key = self._get_s3_key(user_id, metric)
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            try:
                self.s3_client.put_object(Bucket=self.s3_bucket, Key=key, Body=buf.getvalue())
                print(f"Successfully uploaded {metric} data for {user_id} to S3 at key: {key}")
            except Exception as e:
                print(f"Error uploading {metric} data for {user_id} to S3: {e}")
            print(f"Uploaded {metric} data for {user_id} to S3.")

    def generate_dashboard_graph(self, user_id: str) -> io.BytesIO:
        """
        Generates a plot of all metrics for a user.
        """
        plt.figure(figsize=(10, 6))
        all_data_found = False
        for m in METRICS:
            df = self.get_metrics_from_s3(user_id, m)
            print(f"Data fetched for {m} for user {user_id}:")
            print(df.to_string() if not df.empty else "Empty DataFrame") # Print DataFrame content
            if not df.empty:
                all_data_found = True
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Sort by timestamp to ensure correct plotting
                df = df.sort_values(by='timestamp')
                plt.plot(df['timestamp'], df['value'], label=m)

        if all_data_found:
            plt.title(f'Eczema Metrics for {user_id}')
            plt.xlabel('Timestamp')
            plt.ylabel('Severity Level (1-100%)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, "No data available for this user.",
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)
            plt.title(f'Dashboard for {user_id}')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

# --- Flask Application (app.py) ---
# This part would typically be in a separate file (e.g., app.py)
# where you import your EczemaAnalyzer class.

from flask import Flask, request, render_template_string, send_file, jsonify

app = Flask(__name__)

# Initialize your analyzer class
# Ensure GEMINI_API_KEY and S3_BUCKET are set in your environment variables
analyzer = EczemaAnalyzer(S3_BUCKET, GEMINI_API_KEY, GEMINI_API_URL)

@app.route('/', methods=['GET'])
def index():
    return render_template_string("""
      <h1>Eczema Analyzer</h1>
      <form action="/scan" method="post" enctype="multipart/form-data">
        User ID: <input name="user_id" required><br>
        Photo:  <input type="file" name="photo" accept="image/*" required><br>
        <button>Scan</button>
      </form>
    """)

@app.route('/scan', methods=['POST'])
def scan():
    user_id = request.form['user_id']
    file = request.files['photo']
    image_bytes = file.read()

    # Call the class method to get Gemini results
    metrics = analyzer.get_gemini_analysis_results(image_bytes)
    metrics_json = json.dumps(metrics, indent=2)

    # Convert image to base64 for inline preview
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{b64_image}" width="300" alt="Uploaded Image">'

    html = f"""
    <h2>Scan Results for {user_id}</h2>
    <div style="display: flex; gap: 40px;">
      <div>
        <h3>Uploaded Image</h3>
        {img_tag}
      </div>
      <div>
        <h3>Metrics (1–100%)</h3>
        <pre>{metrics_json}</pre>
      </div>
    </div>
    <form action="/save" method="post">
      <input type="hidden" name="user_id" value="{user_id}">
      <input type="hidden" name="metrics" value='{json.dumps(metrics)}'>
      <button type="submit">Save Data</button>
    </form>
    <br>
    <a href="/">Back to Home</a>
    """
    return render_template_string(html)

@app.route('/save', methods=['POST'])
def save():
    uid = request.form['user_id']
    metrics = json.loads(request.form['metrics'])
    # Call the class method to send to AWS (S3)
    analyzer.send_metrics_to_s3(uid, metrics)
    return f"Saved. <a href='/dashboard?user_id={uid}'>View Dashboard</a>"

@app.route('/dashboard')
def dashboard():
    uid = request.args.get('user_id')
    if not uid:
        return "<form><input name='user_id'><button>Go</button></form>"
    return f"<h1>Dashboard {uid}</h1><img src='/graph.png?user_id={uid}'>"

@app.route('/graph.png')
def graph_png():
    uid = request.args.get('user_id')
    # Call the class method to generate the graph (which uses S3 methods internally)
    img_buf = analyzer.generate_dashboard_graph(uid)
    return send_file(img_buf, mimetype='image/png')

if __name__ == '__main__':
    # For local development, ensure these environment variables are set:
    # export S3_BUCKET_NAME="your-s3-bucket-name"
    # export GEMINI_API_KEY="your-gemini-api-key"
    app.run(debug=True, host='0.0.0.0', port=8000) # host='0.0.0.0' to be accessible from network