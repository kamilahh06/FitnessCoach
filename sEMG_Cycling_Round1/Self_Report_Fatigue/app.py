from flask import Flask, request, jsonify
import csv
from datetime import datetime

app = Flask(__name__)
CSV_FILE = "/home/yourusername/FitnessCoach/sEMG_Cycling_Round1/Self_Report_Fatigue/self_report_fatigue.csv"

@app.route('/log', methods=['POST'])
def log_fatigue():
    data = request.get_json()
    fatigue_level = data.get('fatigue', 'Unknown')
    timestamp = datetime.now().isoformat()
    
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, fatigue_level])
    
    return jsonify({"status": "success", "fatigue": fatigue_level})

if __name__ == '__main__':
    app.run()