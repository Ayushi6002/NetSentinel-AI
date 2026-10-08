import os
from pathlib import Path
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / 'frontend'

# Load environment variables from .env if present
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize Flask application
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# Safe reading of Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint required by NetSentinel AI specification."""
    return jsonify({
        "status": "ok",
        "project": "NetSentinel AI",
        "phase": "Phase 1 - Network Traffic Intelligence"
    })


@app.route('/')
def serve_index():
    """Serve the frontend placeholder if available."""
    if (FRONTEND_DIR / 'index.html').exists():
        return send_from_directory(str(FRONTEND_DIR), 'index.html')
    return jsonify({
        "message": "NetSentinel AI Backend is active",
        "health": "/api/health"
    })


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='127.0.0.1', port=port, debug=True)
