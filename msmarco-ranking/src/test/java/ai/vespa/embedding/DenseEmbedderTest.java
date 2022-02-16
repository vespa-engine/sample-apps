package ai.vespa.embedding;

import ai.vespa.TokenizerFactory;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentType;
import com.yahoo.document.TensorDataType;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.search.Query;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;

import java.util.Optional;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;



public class DenseEmbedderTest {

    static final DenseEmbedder embedder;

    static {
        ModelsEvaluator evaluator = ModelsEvaluatorTester.create("src/main/application/models/");
        EmbeddingConfig.Builder builder = new EmbeddingConfig.Builder();
        builder.max_document_length(256).max_query_length(32);
        builder.model_name("dense_encoder");
        embedder = new DenseEmbedder(evaluator,TokenizerFactory.getEmbedder(),builder.build());
    }

    private static DocumentType createDocType(String fieldName, String documentType) {
        DocumentType type = new DocumentType(documentType);
        type.addField(fieldName, DataType.STRING);
        type.addField("text_embedding",
                TensorDataType.getTensor(TensorType.fromSpec("tensor<float>(d0[384])")));
        return type;
    }

    @Test
    public void test_happy_path() {
        Query query = new Query("/?query=what+was+the+project");
        Optional<Tensor> query_embedding = embedder.embed(query);
        Document doc = new Document(createDocType("text","passage"),
                "id:any:passage::0");
        doc.setFieldValue("text", new StringFieldValue("what was the project"));
        Optional<Tensor> doc_embedding = embedder.embed(doc,"text");

        assertFalse(query_embedding.isEmpty());
        assertFalse(doc_embedding.isEmpty());
        double score = query_embedding.get().multiply(doc_embedding.get()).sum().asDouble();
        assertEquals(1.0, score,0.000001);
    }
}
