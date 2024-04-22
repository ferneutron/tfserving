import json
import requests
import pandas as pd


if __name__ == "__main__":

    df = pd.read_csv("test.csv")
    df.drop("target", axis=1, inplace=True)

    headers = {
        "content-type": "application/json"
        }
    
    request = {
        "signature_name": "serving_default", 
        "instances": df.iloc[0:3].values.tolist()
        }
    
    data = json.dumps(request)

    # response = requests.post('http://localhost:8501/v1/models/saved_model/versions/1:predict', data=data, headers=headers)
    response = requests.post('http://localhost:8501/v1/models/saved_model:predict', data=data, headers=headers)

    predictions = json.loads(response.text)['predictions']

    print(f"Predictions: {predictions}")