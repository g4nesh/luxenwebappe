import os
# Set environment variables
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAUMHDSKJ2ZE3IPDNA'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['GEMINI_API_KEY'] = 'AIzaSyDATlzkJ-auty-coYJEkcl1PoJFd1Vj13o'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['S3_BUCKET_NAME'] = 'luxen-test-storage-v1'

from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, os
from werkzeug.utils import secure_filename
import time
import base64
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import io
import uuid
from datetime import datetime
from server_2 import EczemaAnalyzer, S3_BUCKET, GEMINI_API_KEY, GEMINI_API_URL
import json
import urllib.parse
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Add custom Jinja2 filter for JSON parsing
@app.template_filter('from_json')
def from_json(value):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value

# Configure maximum request size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB in bytes

# Configure upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'dcm', 'nii', 'nii.gz', 'jpg', 'jpeg', 'png'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize S3 client
s3 = boto3.client('s3')

# Initialize the EczemaAnalyzer
analyzer = EczemaAnalyzer(S3_BUCKET, GEMINI_API_KEY, GEMINI_API_URL)

print(f"S3_BUCKET: {S3_BUCKET}")
print(f"GEMINI_API_KEY: {GEMINI_API_KEY}")

# Define the fixed S3 folder for all data
S3_FIXED_FOLDER = 'luxenaibusiness@gmail.com/'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    with sqlite3.connect('luxen.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE,
                password TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT,
                result TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                s3_key TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT,
                redness REAL,
                scaling REAL,
                texture REAL,
                color_variation REAL,
                severity REAL,
                predicted_deficiencies TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

def verify_db():
    try:
        with sqlite3.connect('luxen.db') as conn:
            # Check if users table exists and has correct structure
            cursor = conn.execute("PRAGMA table_info(users)")
            columns = {row[1] for row in cursor.fetchall()}
            required_columns = {'id', 'email', 'password'}
            if not required_columns.issubset(columns):
                print("Recreating users table...")
                conn.execute('DROP TABLE IF EXISTS users')
                conn.execute('''
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE,
                        password TEXT
                    )
                ''')
            
            # Check if scans table exists and has correct structure
            cursor = conn.execute("PRAGMA table_info(scans)")
            columns = {row[1] for row in cursor.fetchall()}
            required_columns = {'id', 'user_id', 'filename', 'result', 'upload_date', 's3_key'}
            if not required_columns.issubset(columns):
                print("Recreating scans table...")
                conn.execute('DROP TABLE IF EXISTS scans')
                conn.execute('''
                    CREATE TABLE scans (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        filename TEXT,
                        result TEXT,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        s3_key TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                ''')
            
            # Check if scan_results table exists and has correct structure
            cursor = conn.execute("PRAGMA table_info(scan_results)")
            columns = {row[1] for row in cursor.fetchall()}
            required_columns = {'id', 'user_id', 'timestamp', 'redness', 'scaling', 'texture', 'color_variation', 'severity', 'predicted_deficiencies'}
            if not required_columns.issubset(columns):
                print("Recreating scan_results table...")
                conn.execute('DROP TABLE IF EXISTS scan_results')
                conn.execute('''
                    CREATE TABLE scan_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        timestamp TEXT,
                        redness REAL,
                        scaling REAL,
                        texture REAL,
                        color_variation REAL,
                        severity REAL,
                        predicted_deficiencies TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                ''')
    except Exception as e:
        print(f"Database verification error: {str(e)}")
        raise

# Initialize and verify database
init_db()
verify_db()

def get_user_folder(email):
    """Get the S3 folder path for a user"""
    # URL encode the email to handle special characters
    safe_email = urllib.parse.quote(email, safe='')
    return f"{safe_email}/"

def save_user_to_s3(user_data):
    """Save user data to S3"""
    try:
        email = user_data['email']
        user_folder = get_user_folder(email)
        profile_key = f"{user_folder}profile.json"
        upload_to_s3(json.dumps(user_data).encode(), profile_key)
        return True
    except Exception as e:
        print(f"Error saving user to S3: {str(e)}")
        return False

def get_user_from_s3(email):
    """Get user data from S3"""
    try:
        user_folder = get_user_folder(email)
        profile_key = f"{user_folder}profile.json"
        response = s3.get_object(Bucket=S3_BUCKET, Key=profile_key)
        return json.loads(response['Body'].read().decode())
    except Exception as e:
        print(f"Error getting user from S3: {str(e)}")
        return None

def delete_user_from_s3(email):
    """Delete user data from S3"""
    try:
        user_folder = get_user_folder(email)
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=user_folder
        )
        if 'Contents' in response:
            delete_objects = [{'Key': obj['Key']} for obj in response['Contents']]
            s3.delete_objects(
                Bucket=S3_BUCKET,
                Delete={'Objects': delete_objects, 'Quiet': True}
            )
        return True
    except Exception as e:
        print(f"Error deleting user from S3: {str(e)}")
        return False

def upload_to_s3(file_data, s3_key):
    """Upload file data to S3"""
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=file_data
        )
        return True
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return False

def get_from_s3(s3_key):
    """Get file data from S3"""
    try:
        response = s3.get_object(
            Bucket=S3_BUCKET,
            Key=s3_key
        )
        return response['Body'].read()
    except Exception as e:
        print(f"Error getting from S3: {str(e)}")
        return None

def delete_from_s3(s3_key):
    """Delete file from S3"""
    try:
        s3.delete_object(
            Bucket=S3_BUCKET,
            Key=s3_key
        )
        return True
    except Exception as e:
        print(f"Error deleting from S3: {str(e)}")
        return False

def analyze_scan(file_data, filename):
    """Analyze a scan using Gemini AI and return metrics."""
    try:
        metrics = analyzer.get_gemini_analysis_results(file_data)
        print(f"Raw metrics from analyzer: {json.dumps(metrics, indent=2)}")  # Debug log
        
        # Ensure predicted_deficiencies is a list
        predicted_deficiencies = metrics.get('Predicted Deficiency', [])
        if isinstance(predicted_deficiencies, str):
            try:
                predicted_deficiencies = json.loads(predicted_deficiencies)
            except json.JSONDecodeError:
                predicted_deficiencies = []
        
        formatted_metrics = {
            'redness_level': metrics['Redness Level'],
            'scaling_level': metrics['Scaling Level'],
            'texture_score': metrics['Texture Score'],
            'color_variation': metrics['Color Variation'],
            'severity_score': metrics['Severity Score'],
            'predicted_deficiencies': predicted_deficiencies
        }
        
        print(f"Formatted metrics: {json.dumps(formatted_metrics, indent=2)}")  # Debug log
        
        # Save metrics to database
        with sqlite3.connect('luxen.db') as conn:
            conn.execute('''
                INSERT INTO scan_results 
                (user_id, timestamp, redness, scaling, texture, color_variation, severity, predicted_deficiencies) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session['user_id'],
                datetime.utcnow().isoformat(),
                float(metrics['Redness Level']),
                float(metrics['Scaling Level']),
                float(metrics['Texture Score']),
                float(metrics['Color Variation']),
                float(metrics['Severity Score']),
                json.dumps(predicted_deficiencies)
            ))
        
        # Return the formatted metrics as a JSON string
        return json.dumps(formatted_metrics)
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        default_result = {
            'redness_level': 0,
            'scaling_level': 0,
            'texture_score': 0,
            'color_variation': 0,
            'severity_score': 0,
            'predicted_deficiencies': [{
                'deficiency': 'No specific deficiency pattern detected',
                'score': 0,
                'confidence': 'Low',
                'recommendation': 'For proper assessment of skin conditions and potential deficiencies, please consult a healthcare provider.'
            }]
        }
        return json.dumps(default_result)

def save_scan_data(email, filename, result, file_data=None):
    """Save scan file, analysis result to S3 and scan record to database."""
    try:
        user_folder = get_user_folder(email)
        timestamp = datetime.utcnow().isoformat()

        # Save scan file to S3
        scan_key = f"{user_folder}scans/{filename}"
        if file_data and not upload_to_s3(file_data, scan_key):
            print(f"Error uploading scan file {filename} to S3")
            return False

        # Save analysis result to S3
        analysis_key = f"{user_folder}analysis/{filename}_analysis.json"
        analysis_data = {
            'timestamp': timestamp,
            'filename': filename,
            'result': result
        }
        print(f"Saving analysis data to S3: {json.dumps(analysis_data, indent=2)}")  # Debug log
        if not upload_to_s3(json.dumps(analysis_data).encode(), analysis_key):
            print(f"Error uploading analysis for {filename} to S3")
            return False

        # Get the next scan number for this user
        with sqlite3.connect('luxen.db') as conn:
            scan_count = conn.execute('SELECT COUNT(*) FROM scans WHERE user_id = ?', 
                                    (session['user_id'],)).fetchone()[0]
            scan_name = f"Scan {scan_count + 1}"

            # Save record to database
            conn.execute('INSERT INTO scans (user_id, filename, result, s3_key) VALUES (?, ?, ?, ?)',
                        (session['user_id'], scan_name, result, scan_key))
        return True

    except Exception as e:
        print(f"Error in save_scan_data: {str(e)}")
        return False

def save_metrics_and_graph(redness, scaling, texture, color_variation, severity, predicted_deficiencies):
    """Save metrics to database and generate a time series graph."""
    timestamp = datetime.utcnow().isoformat()
    
    # Save metrics to database
    with sqlite3.connect('luxen.db') as conn:
        conn.execute('''
            INSERT INTO scan_results 
            (timestamp, redness, scaling, texture, color_variation, severity, predicted_deficiencies) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, redness, scaling, texture, color_variation, severity, json.dumps(predicted_deficiencies)))
        
        # Fetch all metrics for this user ordered by timestamp
        results = conn.execute('''
            SELECT timestamp, redness, scaling, texture, color_variation, severity, predicted_deficiencies 
            FROM scan_results 
            ORDER BY timestamp ASC
        ''').fetchall()
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results, columns=['timestamp', 'redness', 'scaling', 'texture', 'color_variation', 'severity', 'predicted_deficiencies'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['redness'], label='Redness', color='#e74c3c', marker='o')
    plt.plot(df['timestamp'], df['scaling'], label='Scaling', color='#2ecc71', marker='o')
    plt.plot(df['timestamp'], df['texture'], label='Texture', color='#3498db', marker='o')
    plt.plot(df['timestamp'], df['color_variation'], label='Color Variation', color='#f39c12', marker='o')
    plt.plot(df['timestamp'], df['severity'], label='Severity', color='#8e44ad', marker='o')
    
    plt.title('Skin Condition Metrics Over Time')
    plt.xlabel('Time')
    plt.ylabel('Score (0-100)')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Upload to S3
    s3_key = f"graphs/scan_report_{timestamp}.png"
    upload_to_s3(buf.getvalue(), s3_key)

def preprocess_image(image_data):
    """Standardize image format and size for consistent analysis."""
    try:
        # Convert to PIL Image
        if isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data))
        else:
            img = image_data

        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Calculate new dimensions while maintaining aspect ratio
        target_width = 800
        target_height = 600
        aspect_ratio = img.width / img.height

        if aspect_ratio > 1:  # Wider than tall
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:  # Taller than wide
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        # Resize image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new image with target dimensions and white background
        new_img = Image.new('RGB', (target_width, target_height), 'white')
        
        # Calculate position to paste resized image (centered)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # Paste resized image onto white background
        new_img.paste(img, (paste_x, paste_y))

        # Convert back to bytes
        img_byte_arr = io.BytesIO()
        new_img.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return image_data  # Return original if preprocessing fails

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        with sqlite3.connect('luxen.db') as conn:
            try:
                cursor = conn.execute('INSERT INTO users (email, password) VALUES (?, ?)',
                                    (email, password))
                user_id = cursor.lastrowid
                user_data = {
                    'id': user_id,
                    'email': email,
                    'password_hash': password,
                    'created_at': datetime.utcnow().isoformat(),
                    'last_login': None,
                    'scans': []
                }
                if save_user_to_s3(user_data):
                    flash('Account created successfully! Please log in.', 'success')
                    return redirect('/login')
                else:
                    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
                    flash('Error creating account. Please try again.', 'error')
                    return redirect('/signup')
            except sqlite3.IntegrityError:
                flash('Email already exists. Please try another email.', 'error')
                return redirect('/signup')
            except Exception as e:
                flash('An unexpected error occurred during signup.', 'error')
                print(f"Signup error: {str(e)}")
                return redirect('/signup')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            with sqlite3.connect('luxen.db') as conn:
                user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
                if user and check_password_hash(user[2], password):
                    session['user_id'] = user[0]
                    session['user_email'] = user[1] # Store email in session
                    user_data = get_user_from_s3(email)
                    if user_data:
                        user_data['last_login'] = datetime.utcnow().isoformat()
                        save_user_to_s3(user_data)
                    return redirect('/dashboard')
                flash('Invalid email or password.', 'error')
                return redirect('/login')
        except Exception as e:
            print(f"Login error: {str(e)}")
            flash('An error occurred during login. Please try again.', 'error')
            return redirect('/login')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session or 'user_email' not in session:
        return redirect('/login')
    
    user_email = session['user_email']

    if request.method == 'POST':
        if 'camera_capture' in request.form:
            try:
                image_data = request.form['camera_capture']
                if not image_data:
                    flash('No image data received.', 'error')
                    return redirect('/dashboard')

                # Remove the data URL prefix if present
                if image_data.startswith('data:image/png;base64,'):
                    image_data = image_data.replace('data:image/png;base64,', '')
                
                # Generate a unique filename with timestamp
                timestamp = int(time.time())
                filename = f"camera_capture_{timestamp}.jpg"  # Changed to .jpg for consistency
                
                try:
                    # Decode base64 image data
                    file_data = base64.b64decode(image_data)
                    
                    # Validate the image data
                    if len(file_data) == 0:
                        raise ValueError("Empty image data received")
                    
                    # Preprocess the image
                    processed_data = preprocess_image(file_data)
                    
                    # Analyze the scan
                    result = analyze_scan(processed_data, filename)
                    
                    # Save the scan data
                    if save_scan_data(user_email, filename, result, file_data=processed_data):
                        flash('Photo captured and analyzed successfully!', 'success')
                    else:
                        flash('Error saving captured photo.', 'error')

                except base64.binascii.Error:
                    flash('Invalid image data received.', 'error')
                except ValueError as ve:
                    flash(f'Error processing image: {str(ve)}', 'error')
                except Exception as e:
                    flash('Error processing captured photo.', 'error')
                    print(f"Camera capture error: {str(e)}")

            except Exception as e:
                flash('Error saving captured photo. Please try again.', 'error')
                print(f"Camera capture error: {str(e)}")
            return redirect('/dashboard')

        if 'scan' not in request.files:
            flash('No file selected.', 'error')
            return redirect('/dashboard')
        
        file = request.files['scan']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect('/dashboard')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = f"{os.path.splitext(filename)[0]}_{int(time.time())}.jpg"  # Changed to .jpg for consistency
            
            try:
                file_data = file.read()
                
                # Preprocess the image
                processed_data = preprocess_image(file_data)
                
                result = analyze_scan(processed_data, filename)
                
                if save_scan_data(user_email, filename, result, file_data=processed_data):
                    flash('File uploaded and analyzed successfully!', 'success')
                else:
                    flash('Error uploading file.', 'error')

            except Exception as e:
                flash('Error uploading file. Please try again.', 'error')
                print(f"Upload error: {str(e)}")
        else:
            flash('Invalid file type. Please upload a supported medical image format.', 'error')
    
    with sqlite3.connect('luxen.db') as conn:
        scans = conn.execute('SELECT * FROM scans WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],)).fetchall()
    return render_template('dashboard.html', scans=scans)

@app.route('/delete_scan', methods=['POST'])
def delete_scan():
    if 'user_id' not in session or 'user_email' not in session:
        return redirect('/login')
    
    scan_id = request.form['scan_id']
    user_id = session['user_id']
    user_email = session['user_email']

    try:
        with sqlite3.connect('luxen.db') as conn:
            # Get S3 key and filename before deleting from DB
            scan = conn.execute('SELECT s3_key, filename FROM scans WHERE id=? AND user_id=?', 
                              (scan_id, user_id)).fetchone()
            
            if scan:
                s3_key = scan[0] # This should be the scan file key (e.g., useremail/scans/filename)
                filename = scan[1]
                user_folder = get_user_folder(user_email)
                
                # Construct the analysis key based on the scan filename
                analysis_key = f"{user_folder}analysis/{filename}_analysis.json"

                # Delete from S3 (scan file and analysis file)
                deleted_scan = delete_from_s3(s3_key)
                deleted_analysis = delete_from_s3(analysis_key)

                if deleted_scan and deleted_analysis:
                    # Delete from database
                    conn.execute('DELETE FROM scans WHERE id=? AND user_id=?', 
                               (scan_id, user_id))
                    flash('Scan and analysis deleted successfully.', 'success')
                elif deleted_scan or deleted_analysis:
                     flash('Partial deletion: Check S3 bucket.', 'warning')
                     # Still delete from DB if at least one S3 file was deleted
                     conn.execute('DELETE FROM scans WHERE id=? AND user_id=?', 
                               (scan_id, user_id))
                else:
                    flash('Error deleting scan from S3.', 'error')
    except Exception as e:
        flash('Error deleting scan. Please try again.', 'error')
        print(f"Delete error: {str(e)}")
    
    return redirect('/dashboard')

@app.route('/report/<int:scan_id>')
def report(scan_id):
    if 'user_id' not in session or 'user_email' not in session:
        return redirect('/login')
    
    user_id = session['user_id']
    user_email = session['user_email']

    with sqlite3.connect('luxen.db') as conn:
        # Get the scan record
        scan = conn.execute('SELECT * FROM scans WHERE id=? AND user_id=?', 
                          (scan_id, user_id)).fetchone()
        if not scan:
            flash('Scan not found.', 'error')
            return redirect('/dashboard')
        
        # Get the latest scan results for this scan
        scan_results = conn.execute('''
            SELECT redness, scaling, texture, color_variation, severity 
            FROM scan_results 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''', (user_id,)).fetchone()
        
        if not scan_results:
            flash('No scan results found.', 'error')
            return redirect('/dashboard')
        
        # Extract metrics
        redness, scaling, texture, color_var, severity = scan_results
        
        # Define absolute thresholds for different conditions
        # These thresholds are based on a 0-100 scale
        is_high_scaling = scaling >= 60
        is_moderate_scaling = scaling >= 40
        is_high_redness = redness >= 60
        is_moderate_redness = redness >= 40
        is_high_texture = texture >= 60
        is_moderate_texture = texture >= 40
        is_high_color_var = color_var >= 60
        is_moderate_color_var = color_var >= 40
        is_severe = severity >= 70
        is_moderate = severity >= 50
        is_mild = severity >= 30

        # Check if overall metrics indicate healthy skin
        is_healthy = all(metric < 30 for metric in [redness, scaling, texture, color_var, severity])

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

        # If metrics indicate healthy skin, give very low scores
        if is_healthy:
            for deficiency in deficiency_scores:
                deficiency_scores[deficiency] = 0.5
        else:
            # Vitamin D deficiency patterns
            if is_high_scaling:
                deficiency_scores["Vitamin D"] += 2
            if is_high_redness:
                deficiency_scores["Vitamin D"] += 1.5
            if is_high_texture:
                deficiency_scores["Vitamin D"] += 1

            # Vitamin B12 deficiency patterns
            if is_high_color_var:
                deficiency_scores["Vitamin B12"] += 2
            if is_high_texture:
                deficiency_scores["Vitamin B12"] += 1.5
            if is_high_redness:
                deficiency_scores["Vitamin B12"] += 1

            # Zinc deficiency patterns
            if is_high_scaling and is_severe:
                deficiency_scores["Zinc"] += 2
            if is_high_texture:
                deficiency_scores["Zinc"] += 1.5
            if is_high_redness:
                deficiency_scores["Zinc"] += 1

            # Essential fatty acids deficiency patterns
            if is_high_scaling and is_high_texture:
                deficiency_scores["Essential Fatty Acids"] += 2
            if is_high_texture and is_mild:
                deficiency_scores["Essential Fatty Acids"] += 1.5
            if is_high_color_var:
                deficiency_scores["Essential Fatty Acids"] += 1

            # Iron deficiency patterns
            if is_high_color_var and is_high_redness:
                deficiency_scores["Iron"] += 2
            if is_high_redness and is_moderate:
                deficiency_scores["Iron"] += 1.5
            if is_high_texture:
                deficiency_scores["Iron"] += 1

            # Vitamin A deficiency patterns
            if is_high_scaling and is_high_texture:
                deficiency_scores["Vitamin A"] += 2
            if is_high_color_var:
                deficiency_scores["Vitamin A"] += 1.5
            if is_moderate_redness:
                deficiency_scores["Vitamin A"] += 1

            # Vitamin E deficiency patterns
            if is_high_texture and is_high_color_var:
                deficiency_scores["Vitamin E"] += 2
            if is_high_scaling:
                deficiency_scores["Vitamin E"] += 1.5
            if is_moderate_redness:
                deficiency_scores["Vitamin E"] += 1

            # Vitamin C deficiency patterns
            if is_high_redness and is_high_texture:
                deficiency_scores["Vitamin C"] += 2
            if is_high_color_var:
                deficiency_scores["Vitamin C"] += 1.5
            if is_moderate_scaling:
                deficiency_scores["Vitamin C"] += 1

        # Sort deficiencies by score and get top 3
        sorted_deficiencies = sorted(deficiency_scores.items(), key=lambda x: x[1], reverse=True)
        top_deficiencies = sorted_deficiencies[:3]

        # Format the results
        scan_result = []
        for deficiency, score in top_deficiencies:
            if score > 0:  # Only include deficiencies with a score
                # More lenient confidence levels
                confidence = "High" if score >= 2.5 else "Medium" if score >= 1.5 else "Low"
                recommendation = {
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
                }.get(deficiency, "Consult a healthcare provider for proper assessment and treatment recommendations.")

                scan_result.append({
                    "deficiency": deficiency,
                    "score": score,
                    "confidence": confidence,
                    "recommendation": recommendation
                })

        if not scan_result or is_healthy:
            scan_result = [{
                "deficiency": "No significant deficiency patterns detected",
                "score": 0,
                "confidence": "Low",
                "recommendation": "Your skin appears to be in a relatively normal condition. However, for proper assessment of skin conditions and potential deficiencies, please consult a healthcare provider."
            }]

    return render_template('report.html', scan=scan, scan_result=scan_result)

@app.route('/api/scan-metrics')
def scan_metrics():
    """API endpoint to get scan metrics for the graph."""
    try:
        with sqlite3.connect('luxen.db') as conn:
            # Fetch all metrics for the current user ordered by timestamp
            results = conn.execute('''
                SELECT timestamp, redness, scaling, texture, color_variation, severity, predicted_deficiencies 
                FROM scan_results 
                WHERE user_id = ?
                ORDER BY timestamp DESC
            ''', (session['user_id'],)).fetchall()
            
            metrics = []
            for i, row in enumerate(results):
                metric = {
                    'timestamp': row[0],
                    'redness': float(row[1]) if row[1] is not None else 0,
                    'scaling': float(row[2]) if row[2] is not None else 0,
                    'texture': float(row[3]) if row[3] is not None else 0,
                    'color_variation': float(row[4]) if row[4] is not None else 0,
                    'severity': float(row[5]) if row[5] is not None else 0,
                    'predicted_deficiencies': row[6] if row[6] is not None else 'Unknown'
                }
                
                # Calculate percentage change if this is not the first scan
                if i < len(results) - 1:
                    prev_row = results[i + 1]
                    metric['redness_change'] = calculate_percentage_change(float(prev_row[1]) if prev_row[1] is not None else 0, float(row[1]) if row[1] is not None else 0)
                    metric['scaling_change'] = calculate_percentage_change(float(prev_row[2]) if prev_row[2] is not None else 0, float(row[2]) if row[2] is not None else 0)
                    metric['texture_change'] = calculate_percentage_change(float(prev_row[3]) if prev_row[3] is not None else 0, float(row[3]) if row[3] is not None else 0)
                    metric['color_variation_change'] = calculate_percentage_change(float(prev_row[4]) if prev_row[4] is not None else 0, float(row[4]) if row[4] is not None else 0)
                    metric['severity_change'] = calculate_percentage_change(float(prev_row[5]) if prev_row[5] is not None else 0, float(row[5]) if row[5] is not None else 0)
                else:
                    metric['redness_change'] = 0
                    metric['scaling_change'] = 0
                    metric['texture_change'] = 0
                    metric['color_variation_change'] = 0
                    metric['severity_change'] = 0
                
                metrics.append(metric)
            
            return jsonify(metrics)
    except Exception as e:
        print(f"Error fetching metrics: {str(e)}")
        return jsonify({'error': 'Failed to fetch metrics'}), 500

def calculate_percentage_change(old_value, new_value):
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 100 if new_value > 0 else 0
    return ((new_value - old_value) / old_value) * 100

@app.route('/api/s3-graphs')
def list_s3_graphs():
    # This route lists global graphs, not tied to a specific user folder structure yet
    try:
        s3 = boto3.client('s3')
        # Assuming graphs are still stored in a global 'graphs/' folder
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="graphs/")
        files = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.png') or key.endswith('.jpg'):
                url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}" # Direct S3 URL might not work if bucket is not public
                # It's better to generate a presigned URL here too
                url = s3.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET, 'Key': key}, ExpiresIn=3600)
                files.append(url)
        return jsonify(files[::-1])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit-scan', methods=['POST'])
def submit_scan():
    if 'user_id' not in session or 'user_email' not in session:
        return redirect('/login')

    try:
        # Get metrics from form data
        redness = float(request.form.get('redness', 0))
        scaling = float(request.form.get('scaling', 0))
        texture = float(request.form.get('texture', 0))
        color_variation = float(request.form.get('color_variation', 0))
        severity = float(request.form.get('severity', 0))
        predicted_deficiencies = request.form.get('predicted_deficiencies', 'Unknown')

        # Save metrics to database and generate graph
        if save_metrics_and_graph(redness, scaling, texture, color_variation, severity, predicted_deficiencies):
            flash('Scan metrics submitted successfully!', 'success')
        else:
            flash('Error saving scan metrics.', 'error')
            
    except ValueError as e:
        flash('Invalid metric values provided.', 'error')
        print(f"Value error in submit-scan: {str(e)}")
    except Exception as e:
        flash('Error processing scan metrics.', 'error')
        print(f"Error in submit-scan: {str(e)}")
    
    return redirect('/dashboard')

@app.route('/bulk_delete', methods=['POST'])
def bulk_delete():
    if 'user_id' not in session or 'user_email' not in session:
        return redirect('/login')
    
    user_id = session['user_id']
    user_email = session['user_email']

    try:
        scan_ids = json.loads(request.form['scan_ids'])
        if not scan_ids:
            flash('No scans selected for deletion.', 'warning')
            return redirect('/dashboard')

        with sqlite3.connect('luxen.db') as conn:
            # Get S3 keys and filenames before deleting from DB
            # Ensure only scans owned by the current user are selected
            placeholders = ','.join('?' * len(scan_ids))
            scans_to_delete = conn.execute(f'SELECT s3_key, filename FROM scans WHERE id IN ({placeholders}) AND user_id=?', 
                                         scan_ids + [user_id]).fetchall()
            
            if not scans_to_delete:
                 flash('No matching scans found for deletion.', 'warning')
                 return redirect('/dashboard')

            objects_to_delete = []
            user_folder = get_user_folder(user_email)

            for scan in scans_to_delete:
                s3_key = scan[0] # Scan file key
                filename = scan[1]
                analysis_key = f"{user_folder}analysis/{filename}_analysis.json"
                objects_to_delete.append({'Key': s3_key})
                objects_to_delete.append({'Key': analysis_key})

            # Delete from S3 in bulk
            if objects_to_delete:
                delete_response = s3.delete_objects(
                    Bucket=S3_BUCKET,
                    Delete={'Objects': objects_to_delete, 'Quiet': True}
                )

                # Check for errors during S3 deletion (Optional, depends on Quiet=True)
                # if 'Errors' in delete_response:
                #    print(f"Partial S3 deletion errors: {delete_response['Errors']}")
                #    flash('Partial deletion from S3. Check logs.', 'warning')

            # Delete from database
            conn.execute(f'DELETE FROM scans WHERE id IN ({placeholders}) AND user_id=?', 
                       scan_ids + [user_id])
            
            flash(f'Successfully deleted {len(scans_to_delete)} scan(s) and their analysis.', 'success')

    except json.JSONDecodeError:
        flash('Invalid scan selection data.', 'error')
    except Exception as e:
        flash('Error deleting scans. Please try again.', 'error')
        print(f"Bulk delete error: {str(e)}")
    
    return redirect('/dashboard')

@app.route('/s3-browser')
def s3_browser():
    if 'user_id' not in session or 'user_email' not in session:
        return redirect('/login')
    
    user_email = session['user_email']
    user_folder = get_user_folder(user_email)

    try:
        # List all objects in the user's folder
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=user_folder)
        files = []
        
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                # Exclude the user's main folder key itself if it appears
                if key == user_folder:
                    continue

                last_modified = obj['LastModified']
                size = obj['Size']
                
                # Determine file type based on the key path within the user folder
                file_type = "Unknown"
                if key == f"{user_folder}profile.json":
                    file_type = "User Profile"
                elif key.startswith(f"{user_folder}scans/"):
                    file_type = "Scan File"
                elif key.startswith(f"{user_folder}analysis/"):
                    file_type = "Analysis Data"
                elif key.startswith(f"{user_folder}graphs/"): # If graphs were user-specific
                     file_type = "Graph"

                # Generate a temporary URL for viewing/downloading
                url = s3.generate_presigned_url('get_object',
                    Params={'Bucket': S3_BUCKET, 'Key': key},
                    ExpiresIn=3600)  # URL expires in 1 hour
                
                files.append({
                    'key': key,
                    'last_modified': last_modified,
                    'size': size,
                    'type': file_type,
                    'url': url
                })

        return render_template('s3_browser.html', files=files)
    except Exception as e:
        flash(f'Error accessing S3 bucket: {str(e)}', 'error')
        print(f"S3 Browser error: {str(e)}")
        return redirect('/dashboard')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=False)  # Set debug=False for production
