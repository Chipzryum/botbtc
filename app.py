from flask import Flask, jsonify, send_from_directory, abort
import os
import json
import zlib

app = Flask(__name__, static_folder='Website', static_url_path='')

# Paths to the results JSON files
COMPRESSED_RESULTS_PATH = os.path.join(app.static_folder, 'backtest_results.json.gz')
REGULAR_RESULTS_PATH = os.path.join(app.static_folder, 'backtest_results.json')

@app.route('/')
def index():
    """Serves the main HTML page."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serves other static files (CSS, JS)."""
    return send_from_directory(app.static_folder, filename)

@app.route('/backtest_data')
def get_backtest_data():
    """Provides the backtest results data from the JSON file."""
    if os.path.exists(COMPRESSED_RESULTS_PATH):
        try:
            with open(COMPRESSED_RESULTS_PATH, 'rb') as f:
                compressed_data = f.read()
                decompressed_data = zlib.decompress(compressed_data, zlib.MAX_WBITS | 16)  # Decompress gzip
                return jsonify(json.loads(decompressed_data))
        except Exception as e:
            print(f"Error reading or decompressing {COMPRESSED_RESULTS_PATH}: {e}")
            abort(500, description="Failed to load backtest data.")
    elif os.path.exists(REGULAR_RESULTS_PATH):
        try:
            with open(REGULAR_RESULTS_PATH, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            print(f"Error reading or parsing {REGULAR_RESULTS_PATH}: {e}")
            abort(500, description="Failed to load backtest data.")
    else:
        print(f"Error: {COMPRESSED_RESULTS_PATH} or {REGULAR_RESULTS_PATH} not found.")
        abort(404, description="Backtest results file not found.")

if __name__ == '__main__':
    # Make sure the server is accessible on your network if needed,
    # otherwise use '127.0.0.1' for local access only.
    app.run(debug=True, host='0.0.0.0', port=5001)  # Using port 5001 to avoid conflicts