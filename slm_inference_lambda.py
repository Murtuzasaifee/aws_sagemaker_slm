import json
import boto3

ENDPOINT = "huggingface-pytorch-tgi-inference-2025-05-10-12-01-08-253"
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
endpoint_name = ENDPOINT

def lambda_handler(event, context):

    query_params = event['queryStringParameters']
    query = query_params['query']
    payload = {
        "inputs": query,
         "parameters": {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1.03
            }
         }


    response = sagemaker_runtime.invoke_endpoint(
        EndpointName = endpoint_name,
        ContentType = "application/json",
        Body = json.dumps(payload)
    )

    predictions = json.loads(response['Body'].read().decode('utf-8'))
    final_result =predictions[0]['generated_text']
    print(final_result)

    return {
        'statusCode': 200,
        'body': json.dumps(final_result)
    }
