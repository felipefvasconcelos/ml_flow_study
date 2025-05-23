from mlflow.deployments import get_deploy_client

app_name = 'prob-loan-sagemaker'
arn = '' # role
image_ecr_uri = '' # ecr
region = 'us-east-1'

model_uri = '/home/felipe/repos/03_mlflow/ml_flow_study/projeto/mlartifacts/1/b2fb981cd68d4f1cbd8bdb34bb5ff733/artifacts/modelo.joblib'


config = dict(
    execution_role_arn=arn, # role do sagemaker
    bucket_name='New-s3-bucket', # nome do bucket
    image_url=image_ecr_uri, # imagem conteiner
    region_name=region,
    archive=False,
    instance_type='ml.m4.xlarge', # máquina q vai ta trabalhando
    instance_count=1,
    synchronous=True,
    timeout_seconds=3600,
    variant_name='prod-variant-2', 
    tags={"trianing_timestamp": "2023-12-05"}
)

client = get_deploy_client('sagemaker')
deploy_client = client.create_deployment(
    app_name,
    model_uri=model_uri,
    flavor='python_function',
    config=config
)

print(f'deploy_client: {deploy_client}')