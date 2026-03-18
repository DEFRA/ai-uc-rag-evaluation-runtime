# ai-uc-rag-evaluation-runtime

This repository contains a technical pattern for implementing LLM as a Judge. LLM-As-A-Judge requires asking an LLM to 
score one response in relation to a ground-truth (the assumed ideal response), using a detailed prompt (rubric).

## Who this pattern is for

## The problem

As part of the Large Language Model (LLM) validation framework we need a process for comparing LLM responses to known 
ground truths, e.g. comparing the answers to questions with the known correct answer or comparing LLM generated 
summaries to the human-written equivalent. This repository aims to provide a reference implementation for how LLM as a
judge can be used on the CDP platform.

## The solution

The flow has three phases:

### Submit Evaluation
A client sends a POST /evaluation request to the FastAPI service. The service persists the evaluation run to the Results
Store, enqueues a message on the SQS queue, then returns a 200 OK response containing a run_id.

### Async / Background Worker
The Listener consumes the message from SQS and loads the evaluation run from the Results Store. It then performs a RAG
search using the query from that run. With the search results in hand, the Listener loops over each model and rubric 
combination — for each iteration it sends the query and results to the LLM Judge for evaluation, receives a score back, 
and stores that result against the run in the Results Store.

### Poll for Result
The client polls for the outcome by sending a GET /evaluation/{run_id} request. FastAPI looks up the run in the Results
Store and returns the score in a 200 OK response.

![Evaluation Flow](docs/images/evaluation-flow.png)

The LLM as a judge is evaluated behind an SQS queue as this is expected to be a long-running process. Currently, 
messages need to be processed within the visibility timeout of the message otherwise they could be reprocessed (30 
seconds by default in CDP). As there currently is only a single listener in practice this is not causing an issue but 
for a real system this may need some more thought.

If an exception is raised whilst running LLM as a judge then the message will be retried after the visibility timeout 
has passed. On CDP by default a message will be retried 3 times before being moved to a DLQ. This will prevent stuck 
messages that keep on being evaluated(This could cause significant cost if the LLM keeps being executed on bedrock in a 
stuck loop). For the same reason care must be taken to not rerun any evaluations that have already been Judged by the 
LLM on retries. This can be seen in the queue_listener.py where a tuple containing the query, rubric, model_key are 
checked to see if they are in the done list.

This repository make use of pydantic-ai and pydantic-evaluation. This library allows us to execute LLM models in Bedrock
and importantly use the inference profiles and guardrails that are required by CDP. See how the models are constructed
in judge_service.py. Guardrails need to be added to the settings. The inference profile used for the bedrock model id.
Finally a profile needs to be set on the bedrock model constructed from the provider and the actual model id being used.
This is so pydantic-ai know internally what model is being used. 

```python
_provider = BedrockProvider(region_name=aws_region)
_settings = BedrockModelSettings(
      bedrock_guardrail_config={
          "guardrailIdentifier": guardrails_id,
          "guardrailVersion": guardrails_version,
          "trace": "enabled",
      }
  )

BedrockConverseModel(
    inference_profile_arn,
    provider=_provider,
    profile=_provider.model_profile(model_id),
    settings=_settings,
)
```
At the time of writing not all models allow a provider to be set and may need to be extended to allow this. An example of
this can be seen with the bedrock embedding model in the ai-uc-rag-evaluation-data project 
app/common/embedding/pydantic_ai.py.

## Licence

THIS INFORMATION IS LICENSED UNDER THE CONDITIONS OF THE OPEN GOVERNMENT LICENCE found at:

<http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3>

The following attribution statement MUST be cited in your products and applications when using this information.

> Contains public sector information licensed under the Open Government license v3

### About the licence

The Open Government Licence (OGL) was developed by the Controller of Her Majesty's Stationery Office (HMSO) to enable
information providers in the public sector to license the use and re-use of their information under a common open
licence.

It is designed to encourage use and re-use of information freely and flexibly, with only a few conditions.
