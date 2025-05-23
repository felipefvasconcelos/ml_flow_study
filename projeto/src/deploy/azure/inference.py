import json
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

workspace_name = 'prob-loan-ws'
workspace_location = 'East US'
resource_group = 'azure_mlops'
subscription_id = '1234'
endpoint_name = 'prob-loan-endpoint'

ml_client = MLClient(DefaultAzureCredential(),
                     subscription_id,
                     resource_group,
                     workspace_name)

df_test = pd.read_csv('/home/felipe/repos/03_mlflow/ml_flow_study/projeto/data/raw/test.csv')
data = {'input_data': df_test.iloc[[0]].to_dict(orient='split')}

with open('file.json', 'w') as f:
    json.dump(data, f)

response = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_name,
    request_file='file.json'
)

print(response)