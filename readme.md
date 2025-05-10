# AWS Lambda with SageMaker Text Generation Endpoint

This repository contains code for deploying and using a text generation model (LaMini-T5-738M) via AWS SageMaker with a Lambda function as the API endpoint.

## Architecture Overview

```
Client Request → Lamda Funcation URL → Lambda Function → SageMaker Endpoint → Generated Response
```

The system uses a Hugging Face text2text-generation model (MBZUAI/LaMini-T5-738M) deployed on a SageMaker endpoint. A Lambda function processes API requests and forwards them to the SageMaker endpoint.

## Components

### 1. SageMaker Endpoint

- **Model**: MBZUAI/LaMini-T5-738M (Text-to-Text Generation)
- **Instance Type**: ml.g5.xlarge
- **Endpoint Name**: huggingface-pytorch-tgi-inference-2025-05-10-12-01-08-253

### 2. Lambda Function

- **Function Name**: [Your Lambda Function Name]
- **Region**: us-east-1
- **Purpose**: Acts as an API handler to process requests and communicate with the SageMaker endpoint

## Setup and Deployment

### Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured
- Python 3.8+
- Boto3 library

### SageMaker Model Deployment

1. Run the SageMaker notebook to install required dependencies:
   ```python
   pip install transformers einops accelerate bitsandbytes
   pip install langchain langchain-community langchain-huggingface
   ```

2. Import the necessary libraries and define the model:
   ```python
   from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
   from langchain_huggingface import HuggingFacePipeline
   import torch
   import json
   import sagemaker
   import boto3
   ```

3. Deploy the model to SageMaker:
   ```python
   # Get SageMaker execution role
   try:
       role = sagemaker.get_execution_role()
   except ValueError:
       iam = boto3.client('iam')
       role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
   
   # Configure the Hugging Face model
   hub = {
       'HF_MODEL_ID': 'MBZUAI/LaMini-T5-738M',
       'HF_TASK': 'text2text-generation',
       'device_map': 'auto',
       'torch_dtype': 'torch.float32'
   }
   
   # Create and deploy the model
   from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
   
   huggingface_model = HuggingFaceModel(
       image_uri=get_huggingface_llm_image_uri("huggingface", version="3.2.3"),
       env=hub,
       role=role,
   )
   
   predictor = huggingface_model.deploy(
       initial_instance_count=1,
       instance_type="ml.g5.xlarge",
       container_startup_health_check_timeout=300,
   )
   ```

4. Test the endpoint directly in SageMaker:
   ```python
   predictor.predict({
       "inputs": "Write an article about Cyber Security",
   })
   ```

### Lambda Function Setup

1. Create a new Lambda function in the AWS Console with Python 3.8+ runtime
2. Copy the provided Lambda code into the function
3. Set the environment variable if needed or hardcode the endpoint name as shown in the example
4. Set the timeout to at least 10 mins
5. Configure appropriate IAM permissions (sagemaker:InvokeEndpoint)
6. (Optional) Set up API Gateway to expose the Lambda function as an API endpoint

## Using the API

### API Request Format

Make a GET request with the following query parameter:

- `query`: The text prompt for the model

Example:
```
https://[your-api-gateway-url]/[stage]/[resource]?query=Write an article about Blockchain and its benefits
```

### API Response Format

The API returns a JSON response with the generated text:

```json
"[Generated text will appear here]"
```

## Configuration Parameters

The Lambda function configures the following generation parameters:

- `max_new_tokens`: 256
- `do_sample`: true
- `temperature`: 0.3
- `top_p`: 0.7
- `top_k`: 50
- `repetition_penalty`: 1.03

You can modify these parameters in the Lambda function code to adjust the generation behavior.

## Troubleshooting

Common issues and solutions:

1. **Timeout Errors**: Increase the Lambda function timeout in the AWS console
2. **Permission Errors**: Ensure the Lambda execution role has `sagemaker:InvokeEndpoint` permissions
3. **Cold Start Delays**: The first request may take longer due to Lambda cold starts
4. **Memory Issues**: Increase Lambda memory allocation if needed

## Cost Considerations

- SageMaker ml.g5.xlarge instances incur charges while running
- Lambda function executions are charged based on execution time and memory usage
- Consider setting up auto-scaling for the SageMaker endpoint during high-demand periods


