import json
import os

LOG_FILE = "inspection_logs.json"


def compute_analytics():

    if not os.path.exists(LOG_FILE):
        return {
            "runs": 0,
            "total_detections": 0,
            "severity_distribution": {}
        }

    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    total_runs = len(logs)

    total_detections = 0
    severity_distribution = {}

    for entry in logs:

        total_detections += entry.get("total_detections", 0)

        severity = entry.get("severity_summary", {})

        for k, v in severity.items():

            severity_distribution[k] = severity_distribution.get(k, 0) + v

    return {
        "runs": total_runs,
        "total_detections": total_detections,
        "severity_distribution": severity_distribution
    }