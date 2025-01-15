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
import java.util.Random;
import java.util.logging.Logger;
import java.util.logging.Level;

public class MyEmbedder implements Embedder {

    private static final Logger logger = Logger.getLogger(MyEmbedder.class.getName());
    private static final int FIXED_DIMENSION_SIZE = 128;
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
        // For this example, we assume the tensor type is something like tensor<float>(d0[128])
        // We'll fill that with random floats. Adjust as needed.
        validateTensorType(tensorType);
        // Log that tensor type is valid
        logger.log(Level.INFO, "Tensor type is valid: " + tensorType.toString());
        // Build a tensor of the given type
        Builder builder = Tensor.Builder.of(tensorType);

        // We assume a single bounded dimension (e.g. d0[128])
        // Fill with random float values
        for (int i = 0; i < FIXED_DIMENSION_SIZE; i++) {
            builder.cell(random.nextFloat(), i);
        }

        return builder.build();
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