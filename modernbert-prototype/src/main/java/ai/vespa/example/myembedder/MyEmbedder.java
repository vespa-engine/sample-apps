package ai.vespa.example.myembedder;

import com.yahoo.language.Language;
import com.yahoo.language.process.Embedder;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.Tensor.Builder;
import com.yahoo.config.FileReference;
import com.google.inject.Inject;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.logging.Logger;
import java.util.logging.Level;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import java.time.Duration;
import java.nio.charset.StandardCharsets;
import com.fasterxml.jackson.databind.ObjectMapper;

public class MyEmbedder implements Embedder {

    private static final Logger logger = Logger.getLogger(MyEmbedder.class.getName());
    private static final int FIXED_DIMENSION_SIZE = 128;
    private static final int MAX_RETRIES = 3;
    private static final Duration TIMEOUT = Duration.ofSeconds(10);
    
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String baseUrl;
    private final Random random = new Random();
    private final MyEmbedderConfig config;
    private final Path modelPath;    
    private final Path vocabPath;    
    private final String modelId;
    

    /**
     * Constructor for MyEmbedder.
     * 
     * @param config the configuration object injected by the container.
     */

    @Inject
    public MyEmbedder(MyEmbedderConfig config) {
        this.config = config;
        
        // Validate required config
        if (config.model().path() == null) {
            throw new IllegalArgumentException("model.path must be specified in config");
        }
        if (config.vocab().path() == null) {
            throw new IllegalArgumentException("vocab.path must be specified in config");
        }
        if (config.model().modelId().isEmpty()) {
            throw new IllegalArgumentException("model.modelId must be specified in config");
        }

        this.modelPath = config.model().path();    
        this.vocabPath = config.vocab().path(); 
        this.modelId = config.model().modelId();
        
        // Log config values
        logger.log(Level.INFO, "MyEmbedder initialized with modelPath: " + modelPath.toString() +
                               ", vocabPath: " + vocabPath.toString() +
                               ", modelId: " + modelId);

        this.baseUrl = "http://embedder:1337";
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(TIMEOUT)
            .version(HttpClient.Version.HTTP_1_1)  // Force HTTP 1.1
            .build();
        this.objectMapper = new ObjectMapper();
        
        initializeModel();
    }

    private void initializeModel() {
        for (int i = 0; i < MAX_RETRIES; i++) {
            try {
                // this is what we need
                HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/models"))
                    .header("Content-Type", "application/json")
                    .GET()
                    .build();
                    
                logger.log(Level.INFO, "Attempting to initialize model (attempt " + (i+1) + "/" + MAX_RETRIES + ")");
                
                HttpResponse<String> response = httpClient.send(request, 
                    HttpResponse.BodyHandlers.ofString());
                    
                if (response.statusCode() == 200) {
                    logger.log(Level.INFO, "Model initialized successfully");
                    // also log response body
                    logger.log(Level.INFO, "Response body: " + response.body());
                    return;
                }
                
                Thread.sleep(2000); // Wait 2 seconds between retries
                
            } catch (Exception e) {
                logger.log(Level.WARNING, "Failed to initialize model (attempt " + (i+1) + "/" + MAX_RETRIES + ")", e);
                if (i == MAX_RETRIES - 1) {
                    throw new RuntimeException("Failed to initialize model after " + MAX_RETRIES + " attempts", e);
                }
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    @Override
    public List<Integer> embed(String text, Context context) {
        // Rest of the existing code
        int tokenCount = 10;
        List<Integer> tokens = new ArrayList<>(tokenCount);
        for (int i = 0; i < tokenCount; i++) {
            tokens.add(random.nextInt(10000));
        }
        // Log that tokens are generated
        logger.log(Level.INFO, "Generated tokens: " + tokens.toString());
        return tokens;
    }

    @Override
    public Tensor embed(String text, Context context, TensorType tensorType) {
        validateTensorType(tensorType);
        logger.log(Level.INFO, "Generating embedding for text: " + text);

        try {
            // Prepare request payload
            // need to match this format:
            // {
            //     "model": "michaelfeil/bge-small-en-v1.5",
            //     "encoding_format": "float",
            //     "input": [
            //       "this is my string to encode"
            //     ],
            //     "modality": "text"
            //   }
            String payload = objectMapper.writeValueAsString(Map.of(
                "model", "nomic-ai/modernbert-embed-base", // modelId,
                "encoding_format", "float",
                "input", List.of(text),
                "modality", "text"
            ));

            // Create and send HTTP request
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/embeddings"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(payload))
                .build();
                
            HttpResponse<String> response = httpClient.send(request, 
                HttpResponse.BodyHandlers.ofString());
                
            if (response.statusCode() != 200) {
                throw new RuntimeException("Embedding failed: " + response.body());
            }

            // Parse response 
            // {
            // "object": "list",
            // "data": [
            //     {
            //     "object": "embedding",
            //     "embedding": [
            //         -0.0686255693435669,
            //         -0.05378103256225586,
            //         -0.04283014312386513,
            //         ... 
            Map<String, Object> jsonResponse = objectMapper.readValue(response.body(), Map.class);
            List<Map<String, Object>> data = (List<Map<String, Object>>) jsonResponse.get("data");
            List<Number> embeddingsList = (List<Number>) data.get(0).get("embedding");
            
            float[] embeddings = new float[embeddingsList.size()];
            for (int i = 0; i < embeddingsList.size(); i++) {
                embeddings[i] = embeddingsList.get(i).floatValue();
            }
            
            // Build tensor
            Builder builder = Tensor.Builder.of(tensorType);
            for (int i = 0; i < FIXED_DIMENSION_SIZE; i++) {
                builder.cell(embeddings[i], i);
            }
            Tensor result = builder.build();
            logger.log(Level.INFO, "Successfully generated embedding tensor");
            return result;
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Failed to generate embedding", e);
            throw new RuntimeException("Failed to generate embedding", e);
        }
    }

    /**
     * Optionally validate the tensor type to ensure it has the
     * shape your embedder expects (e.g. one dimension of size 128).
     */
    private void validateTensorType(TensorType tensorType) {
        // Check only for single dimension with correct size
        if (tensorType.dimensions().size() != 1 ||
            !tensorType.dimensions().get(0).size().isPresent() ||
            tensorType.dimensions().get(0).size().get() != FIXED_DIMENSION_SIZE) {
            throw new IllegalArgumentException("MyEmbedder requires a tensor type with a single " +
                                             "dimension of size " + FIXED_DIMENSION_SIZE +
                                             ", but got: " + tensorType.toString());
        }
    }

    /**
     * Override decode if you want to implement token -> string.
     * The default throws UnsupportedOperationException, which is fine for a basic example.
     */
    // @Override
    // public String decode(List<Integer> tokens, Context context) {
    //     // Implement if needed
    // }

}