
# Models are downloaded from HuggingFace
download-models:
	mkdir -p models/answerai-colbert-small-v1/1
	wget -O models/answerai-colbert-small-v1/1/model.onnx https://huggingface.co/answerdotai/answerai-colbert-small-v1/resolve/main/vespa_colbert.onnx

	mkdir -p models/colbert-v2/1
	wget -O models/colbert-v2/1/model.onnx https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.onnx

# Container image is downloaded from NVIDIA NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
# --model-control-mode=EXPLICIT allows loading, unloading of models with API request: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html#model-control-mode-explicit
run-server:
	podman run --rm \
		-p 8000:8000 -p 8001:8001 -p 8002:8002 \
		-v ${PWD}/models:/models nvcr.io/nvidia/tritonserver:25.03-py3 \
		tritonserver \
		--model-repository=/models \
		--model-control-mode=EXPLICIT

health:
	curl -v localhost:8000/v2/health/ready

run-client:
	cd triton-java-grpc-client && \
	mvn clean package && \
	mvn exec:java -Dexec.mainClass=clients.ColbertGrpcClient