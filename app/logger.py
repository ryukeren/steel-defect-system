import json
import os
from datetime import datetime

LOG_FILE = "inspection_logs.json"


def save_log(data):

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "total_detections": data.get("total_detections", 0),
        "severity_summary": data.get("severity_summary", {}),
        "processing_time_ms": data.get("processing_time_ms", 0)
    }

    if os.path.exists(LOG_FILE):

        with open(LOG_FILE, "r") as f:
            logs = json.load(f)

    else:
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)