import requests
import csv
from datetime import datetime

def fetch_tfl_predictions(line='central'):
    url = f'https://api.tfl.gov.uk/Line/{line}/Arrivals'
    try:
        return requests.get(url).json()
    except Exception as e:
        print("Failed to fetch:", e)
        return []

def log_train_predictions():
    data = fetch_tfl_predictions()
    now = datetime.utcnow().isoformat()
    records=[]

    for item in data:
        records.append({
            'timestamp': now,
            'station': item.get('stationName'),
            'platform': item.get('platformName'),
            'line': item.get('lineName'),
            'destination' : item.get('destinationName'),
            'predictedArrival': item.get('expectedArrival'),
            'vehicledId': item.get('vehicleId'),
            'towards': item.get('towards')
        })

    with open ('./data/central_train_log.csv','a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames = records[0].keys())
        if f.tell() ==0:
            writer.writeheader()
        writer.writerows(records)

if __name__ == "__main__":
    log_train_predictions()
    print("Train predictions logged successfully.")