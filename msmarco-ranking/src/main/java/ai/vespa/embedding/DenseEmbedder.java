package ai.vespa.embedding;

import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.Model;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.google.inject.Inject;
import com.yahoo.component.AbstractComponent;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.datatypes.FieldValue;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.language.process.Embedder;
import com.yahoo.language.wordpiece.WordPieceEmbedder;
import com.yahoo.search.Query;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.functions.Reduce;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Dense single-representation embedder component which can
 * be injected and used by searchers and document processors.
 *
 */

public class DenseEmbedder extends AbstractComponent {

    private final WordPieceEmbedder wordPieceEmbedder;
    private final EmbeddingConfig config;
    private final Model model;
    private static final Tensor BATCH_TENSOR = Tensor.from("tensor<float>(d0[1]):[1]");
    private int documentMaxLength;
    private int queryMaxLength;

    @Inject
    public DenseEmbedder(ModelsEvaluator evaluator,
                         WordPieceEmbedder wordPieceEmbedder, EmbeddingConfig config) {
        this.wordPieceEmbedder = wordPieceEmbedder;
        this.config = config;
        this.model = evaluator.requireModel(this.config.model_name());
        this.queryMaxLength = this.config.max_query_length();
        this.documentMaxLength = this.config.max_document_length();
    }

    /**
     * Create a dense tensor representation of the query by invoking
     * the dense embedding model.
     * @param query the query
     * @return Optional Tensor
     */

    public Optional<Tensor> embed(Query query) {
        String inputQuery = query.getModel().getQueryString();
        if(inputQuery == null)
            return Optional.empty();
        List<Integer> tokens = wordPieceEmbedder.embed(inputQuery,
                new Embedder.Context("q"));
        return Optional.of(getEmbedding(tokens,queryMaxLength));
    }

    /**
     * Create a dense tensor representation of the document by invoking
     * the dense embedding model.
     *
     * The fields are concatenated using the SEP, in the order which they are invoked
     * Possible truncated by the model's max length
     *
     * @param document
     * @param fieldNames the names to use when encoding, most important fields first.
     * @return
     */

    public Optional<Tensor> embed(Document document, String ...fieldNames) {
        var n = fieldNames.length;
        var tokens = new ArrayList<Integer>();
        for(var f:fieldNames) {
            FieldValue text = document.getFieldValue(f);
            if(text == null || text.getDataType() != DataType.STRING)
                continue;
            List<Integer> fieldTokens = wordPieceEmbedder.embed(((StringFieldValue)text).getString(),
                    new Embedder.Context("d"));
            tokens.addAll(fieldTokens);
        }
        return Optional.of(getEmbedding(tokens,this.documentMaxLength));
    }

    public Optional<Tensor> embed(List<Integer> tokens,int maxLength)  {
        return Optional.of(getEmbedding(tokens,maxLength));
    }

    private Tensor getEmbedding(List<Integer> tokens, int maxLength)  {
        int CLS_TOKEN_ID = 101; // [CLS]
        int SEP_TOKEN_ID = 102; // [SEP]
        tokens = tokens.size() > maxLength ? tokens.subList(0,maxLength-2): tokens;
        List<Integer> inputIds = new ArrayList<>(tokens.size()+2);
        inputIds.add(CLS_TOKEN_ID);
        inputIds.addAll(tokens);
        inputIds.add(SEP_TOKEN_ID);

        Tensor input_sequence = getTensorRepresentation(inputIds,"d1");
        Tensor attentionMask = createAttentionMask(input_sequence);
        return removeBatch(this.model.evaluatorOf().
                bind("input_ids",input_sequence.multiply(BATCH_TENSOR)).
                bind("attention_mask",attentionMask.multiply(BATCH_TENSOR)).evaluate());
    }

    private IndexedTensor getTensorRepresentation(List<Integer> input, String dimension)  {
        int size = input.size();
        TensorType type = new TensorType.Builder(TensorType.Value.FLOAT).
                indexed(dimension, size).build();
        IndexedTensor.Builder builder = IndexedTensor.Builder.of(type);
        for (int i = 0; i < size; ++i) {
            builder.cell(input.get(i), i);
        }
        return builder.build();
    }

    private static Tensor createAttentionMask(Tensor d)  {
        return d.map((x) -> x > 0 ? 1:0);
    }

    private static Tensor removeBatch(Tensor embedding) {
        Tensor t = embedding.reduce(Reduce.Aggregator.min, "d0");
        return t.rename("d1","d0");
    }
}
