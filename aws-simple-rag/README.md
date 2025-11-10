<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Retrieval Augmented Generation (RAG) in Vespa using AWS Bedrock models

This sample application demonstrates an end-to-end Retrieval Augmented
Generation application in Vespa, leveraging [AWS Bedrock](https://aws.amazon.com/bedrock/) hosted models.

This sample application focuses on the generation part of RAG, and builds upon
the [MS Marco passage
ranking](https://github.com/vespa-engine/sample-apps/tree/master/msmarco-ranking)
sample application. Please refer to that sample application for details on more
advanced forms of retrieval, such as vector search and cross-encoder
re-ranking. The generation steps in this sample application happen after
retrieval, so the techniques there can easily be used in this application as
well. For the purposes of this sample application, we will use a simple example of [hybrid search and ranking](https://docs.vespa.ai/en/tutorials/hybrid-search.html#hybrid-ranking) to demonstrate Vespa capabilities.

For more details on using retrieval augmented generation in Vespa, please refer to
the [RAG in Vespa](https://docs.vespa.ai/en/llms-rag.html) documentation page.
For more on the general use of LLMs in Vespa, please refer to [LLMs in
Vespa](https://docs.vespa.ai/en/llms-in-vespa.html).

## AWS Setup

### Choose your model

This integration relies on the ability to invoke LLM endpoints with an [OpenAI chat completion API](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html) from Vespa. At the time of writing, the only AWS Bedrock models which can be invoked with the OpenAI Chat completions API are the OpenAI models `gpt-oss-20b`  and  `gpt-oss-120b`.

If you want to use another model, an alternate way is to expose an OpenAI chat completions endpoint through a [Bedrock access gateway](https://github.com/aws-samples/bedrock-access-gateway). The same integration instructions apply after creating the endpoint.

### Choose your region

Availability of the models may vary per region. The format of the bedrock runtime endpoint is as follows:

`https://bedrock-runtime.{region}.amazonaws.com`

You may want to collocate your model endpoint with the AWS region where 
Vespa is deployed. The default Vespa zone where this application will be deployed is in `dev` environment in `aws-us-east-1` region.

### Set-up an AWS Bedrock API Key

Create an [AWS Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html).

### Test your endpoint

You can test your endpoint from curl:

<pre>
export AWS_BEARER_TOKEN_BEDROCK=ABSKQmVk....
curl -X POST https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK" \
  -d '{
   "model": "openai.gpt-oss-20b-1:0",
   "messages": [
       {
           "role": "user",
           "content": "Hello! How are you today?"
       }
   ]
}'
</pre>

Once this test completes successfully, you can proceed to next step.

## Vespa setup

The following is a quick start recipe for getting started with a tiny slice of
the [MS Marco](https://microsoft.github.io/msmarco/) passage ranking dataset to showcase a RAG pattern leveraging AWS Bedrock models.

Please follow the instructions in the [MS Marco passage
ranking](https://github.com/vespa-engine/sample-apps/tree/master/msmarco-ranking) sample
application for instructions on downloading the entire dataset.

In the following we will deploy the sample application to Vespa Cloud.

Make sure that [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html) is
installed. Update to the newest version:
<pre>
$ brew install vespa-cli
</pre>

Download this sample application:
<pre data-test="exec">
$ vespa clone aws-simple-rag bedrock-rag && cd bedrock-rag
</pre>


### Deploying to Vespa Cloud 

Deploy the sample application to Vespa Cloud. Note that this application can fit within the free quota, so it is free to try.

In the following section, we will set the Vespa CLI target to the cloud.
Make sure you have created a tenant at
[console.vespa-cloud.com](https://console.vespa-cloud.com/). Make a note of the
tenant's name; it will be used in the next steps. For more information, see the
Vespa Cloud [getting started](https://cloud.vespa.ai/en/getting-started) guide.

Add your AWS Bedrock API key to the Vespa secret store as described in
[Secret Management](https://cloud.vespa.ai/en/security/secret-store.html#secret-management).
Unless you already have one, create a new vault, and add your AWS Bedrock API key as a secret.

The `services.xml` file must refer to the newly added secret in the secret store.
Replace `<my-vault-name>` and `<my-secret-name>` below with your own values:

```xml
<secrets>
    <bedrock-api-key vault=">my-vault-name>" name="<my-secret-name>"/>
</secrets>
```

Configure the vespa client. Replace `tenant-name` below with your tenant name.
We use the application name `aws-app` here, but you are free to choose your own
application name:
<pre>
$ vespa config set target cloud
$ vespa config set application tenant-name.aws-app
</pre>

Log in and add your public certificates to the application for Dataplane access:
<pre>
$ vespa auth login
$ vespa auth cert
</pre>

Grant application access to the secret.
Applications must be created first so one can use the Vespa Cloud Console to grant access.
The easiest way is to deploy, which will auto-create the application.
The first deployment will fail:

<pre>
$ vespa deploy --wait 900
</pre>

```
[09:47:43] warning Deployment failed: Invalid application: Vault 'my_vault' does not exist,
or application does not have access to it
```

At this point, open the console
(the link is like https://console.vespa-cloud.com/tenant/mytenant/account/secrets)
and grant access:

![edit application access dialog](ext/edit-app-access.png)

Deploy the application again. This can take some time for all nodes to be provisioned:
<pre>
$ vespa deploy --wait 900
</pre>

The application should now be deployed!

### Feeding

Let's feed the documents:
<pre data-test="exec">
$ vespa feed ext/docs.jsonl
</pre>

### Querying: Hybrid Retrieval


Run a query first to check the retrieval:
<pre data-test="exec" data-test-assert-contains="Manhattan">
$ vespa query \
    'yql=select * from passage where ({targetHits:10}userInput(@query)) or ({targetHits:10}nearestNeighbor(embedding,e))' \
    'query=What is the Manhattan Project' \
    'input.query(e)=embed(@query)' \
    hits=3 \
    language=en \
    ranking=hybrid
</pre>


### RAG with AWS Bedrock

To test generation using the OpenAI client, post a query that runs the `bedrock` search chain:
<pre>
$ vespa query \
  'yql=select * from passage where ({targetHits:10}userInput(@query)) or ({targetHits:10}nearestNeighbor(embedding,e))' \
  'query=What is the Manhattan Project' \
  'input.query(e)=embed(@query)' \
   hits=3 \
  language=en \
  ranking=hybrid \
  searchChain=bedrock \
  format=sse \
  traceLevel=1
 </pre> 

Here, we specifically set the search chain to `bedrock`.
This calls the
[RAGSearcher](https://github.com/vespa-engine/vespa/blob/master/container-search/src/main/java/ai/vespa/search/llm/RAGSearcher.java)
which is set up to use the
[OpenAI](https://github.com/vespa-engine/vespa/blob/master/model-integration/src/main/java/ai/vespa/llm/clients/OpenAI.java) client, as we are leveraging the [AWS Bedrock OpenAI chat completions API endpoint](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html).
Note that this requires the AWS Bedrock API key.
We also add a timeout as token generation can take some time.


### Structured output

You can also specify a structured output format for the LLM.
In the example below, we provide a JSON schema to force the LLM to return the answer in 3 different
formats:

- `answer-short`: a short answer to the question
- `answer-short-french`: a translation of the short answer in French
- `answer-short-eli5`: an explanation of the answer as if the user was 5 years old

<pre data-test="exec" data-test-assert-contains="answer-short-french">
$ vespa query \
  'yql=select * from passage where ({targetHits:10}userInput(@query)) or ({targetHits:10}nearestNeighbor(embedding,e))' \
  'query=What is the Manhattan Project' \
  'input.query(e)=embed(@query)' \
   hits=3 \
  language=en \
  ranking=hybrid \
  searchChain=bedrock \
  format=sse \
  llm.json_schema="{\"type\":\"object\",\"properties\":{\"answer-short\":{\"type\":\"string\"},\"answer-short-french\":{\"type\":\"string\",\"description\":\"exact translation of short answer in French language\"},\"answer-short-eli5\":{\"type\":\"string\",\"description\":\"explain the answer like I am 5 years old\"}},\"required\":[\"answer-short\",\"answer-short-french\",\"answer-short-eli5\"],\"additionalProperties\":false}" \
  traceLevel=1
</pre>

The `llm.json_schema` parameter is used to specify the expected output format of the LLM.
The schema is defined in JSON Schema format, which allows you to specify the expected structure of the output.

## Query parameters

The parameters here are:

- `query`: the query used both for retrieval and the prompt question.
- `hits`: the number of hits that Vespa should return in the retrieval stage
- `searchChain`: the search chain set up in `services.xml` that calls the
  generative process
- `format`: sets the format to server-sent events, which will stream the tokens
  as they are generated.
- `traceLevel`: outputs some debug information, such as the actual prompt that
  was sent to the LLM and token timing.

For more information on how to customize the prompt, please refer to the [RAG
in Vespa](https://docs.vespa.ai/en/llms-rag.html) documentation.


## Shutdown and remove the RAG application


To remove the application from Vespa Cloud:
<pre>
$ vespa destroy
</pre>
