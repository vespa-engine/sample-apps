
package clients;


import java.nio.ByteOrder;

import inference.GRPCInferenceServiceGrpc;
import inference.GrpcService;
import io.grpc.ManagedChannelBuilder;

public class ColbertGrpcClient {
	public static void main(String[] args) {
		var host = args.length > 0 ? args[0] : "localhost";
		var port = args.length > 1 ? Integer.parseInt(args[1]) : 8001;
        var modelName = args.length > 2 ? args[2] : "colbert-v2";

		// # Create gRPC stub for communicating with the server
		var channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
        var grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);
        
        // Model loading explained here: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html
        var modelLoadRequest = GrpcService.RepositoryModelLoadRequest.newBuilder().setModelName(modelName).build();
        grpc_stub.repositoryModelLoad(modelLoadRequest);
        
		// check server is live
		var serverLiveRequest = GrpcService.ServerLiveRequest.getDefaultInstance();
		var serverLiverResponse = grpc_stub.serverLive(serverLiveRequest);
		System.out.println(serverLiverResponse);

		// Generate the request
		var request = GrpcService.ModelInferRequest.newBuilder();
		request.setModelName(modelName);
		request.setModelVersion("");

        // Define query text to get embeddings for
        String queryText = "What is machine learning?";
        System.out.println("Query: " + queryText);

        // For ColBERT, we typically need to send tokenized inputs
        // This is a simplified example - in production, use a proper tokenizer
        int[] queryTokenIds = getSimpleTokenIds(queryText);
        var maxQueryLength = 32;  // Adjust based on your model's requirements
        var paddedLength = Math.min(queryTokenIds.length, maxQueryLength);

        // Create tensor contents for input
        var inputData = GrpcService.InferTensorContents.newBuilder();
        
        for (int i = 0; i < paddedLength; i++) {
            inputData.addInt64Contents(queryTokenIds[i]);
        }
        
        // Pad if necessary
        for (int i = paddedLength; i < maxQueryLength; i++) {
            inputData.addInt64Contents(0); // padding token
        }

        // Create input tensor
        var input = GrpcService.ModelInferRequest.InferInputTensor.newBuilder();
        input.setName("input_ids");
        input.setDatatype("INT64");
        input.addShape(1);  // batch size
        input.addShape(maxQueryLength);  // sequence length
        input.setContents(inputData);
        request.addInputs(input);

         // Add attention mask input
         var attentionMaskData = GrpcService.InferTensorContents.newBuilder();
         for (int i = 0; i < paddedLength; i++) {
             attentionMaskData.addInt64Contents(1);  // 1 for actual tokens
         }

         for (int i = paddedLength; i < maxQueryLength; i++) {
             attentionMaskData.addInt64Contents(0);  // 0 for padding
         }

         var attentionMask = GrpcService.ModelInferRequest.InferInputTensor.newBuilder();
         attentionMask.setName("attention_mask");
         attentionMask.setDatatype("INT64");
         attentionMask.addShape(1);  // batch size
         attentionMask.addShape(maxQueryLength);  // sequence length
         attentionMask.setContents(attentionMaskData);
         request.addInputs(attentionMask);

        var response = grpc_stub.modelInfer(request.build());
        System.out.println("Received response with " + response.getOutputsCount() + " outputs");

        // Process the embeddings from the response
        // ColBERT typically returns float embeddings
        if (response.getOutputsCount() > 0) {
            // Get the embedding data - assuming it's returned as raw floating point data
            try {
                float[] embeddings = toFloatArray(response.getRawOutputContentsList().get(0)
                    .asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer());
                
                // Print the first few values of the embeddings
                System.out.println("Embedding dimensions: " + embeddings.length);
                System.out.print("First 5 values: ");
                for (int i = 0; i < Math.min(5, embeddings.length); i++) {
                    System.out.print(embeddings[i] + " ");
                }
                System.out.println();
            } catch (Exception e) {
                System.err.println("Error processing embeddings: " + e.getMessage());
                e.printStackTrace();
            }
        }

        channel.shutdownNow();
	}

        // Helper method to convert FloatBuffer to float array
    public static float[] toFloatArray(java.nio.FloatBuffer buffer) {
        buffer.rewind();
        float[] arr = new float[buffer.remaining()];
        buffer.get(arr);
        return arr;
    }

    // Simple tokenization for demonstration - in production, use a proper tokenizer
    public static int[] getSimpleTokenIds(String text) {
        // This is a very simplified tokenization - you would typically use
        // a proper BERT or transformers tokenizer
        String[] words = text.toLowerCase().replaceAll("[^a-z0-9 ]", "").split("\\s+");
        int[] tokenIds = new int[words.length + 2]; // +2 for [CLS] and [SEP]
        
        // Add [CLS] token (let's use 101 as in BERT)
        tokenIds[0] = 101;
        
        // Simple word-to-id mapping (in reality, use a proper vocabulary)
        for (int i = 0; i < words.length; i++) {
            // Just hash the word to get a token ID (for demonstration only)
            tokenIds[i+1] = Math.abs(words[i].hashCode() % 30000) + 1000; 
        }
        
        // Add [SEP] token (let's use 102 as in BERT)
        tokenIds[words.length + 1] = 102;
        
        return tokenIds;
    }

}
