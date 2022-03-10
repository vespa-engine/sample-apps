// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.docproc;

import ai.vespa.TokenizerFactory;
import ai.vespa.embedding.EmbeddingConfig;
import ai.vespa.embedding.DenseEmbedder;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.*;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.TensorType;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class DenseDocumentEmbedderTest {

    static final DenseEmbedder embedder;

    static {
        ModelsEvaluator evaluator = ModelsEvaluatorTester.create("src/main/application/models/");
        EmbeddingConfig.Builder builder = new EmbeddingConfig.Builder();
        builder.max_document_length(256).max_query_length(32);
        builder.model_name("dense_encoder");
        embedder = new DenseEmbedder(evaluator, TokenizerFactory.getEmbedder(), builder.build());
    }

    private static Processing getProcessing(DocumentOperation... operations) {
        Processing processing = new Processing();
        for (DocumentOperation op : operations) {
            processing.addDocumentOperation(op);
        }
        return processing;
    }

    private static DocumentType createDocType() {
        DocumentType type = new DocumentType("passage");
        type.addField("title",DataType.STRING);
        type.addField("text", DataType.STRING);
        type.addField("mini_document_embedding",
                TensorDataType.getTensor(TensorType.fromSpec("tensor<float>(d0[384])")));
        return type;
    }

    @Test
    public void test_happy_path() throws Exception {
        Document doc = new Document(createDocType(), "id:msmarco:passage::0");
        doc.setFieldValue("text", new
                StringFieldValue("Britney Jean Spears (born December 2, 1981) is an American singer."));
        Processing p = getProcessing(new DocumentPut(doc));
        DocumentProcessor docproc = new DenseDocumentEmbedder(embedder);
        docproc.process(p);
        TensorFieldValue embedding = (TensorFieldValue) doc.getFieldValue("mini_document_embedding");
        assertNotNull(embedding, "Embedding should not be null");

        doc = new Document(createDocType(), "id:msmarco:passage::0");
        doc.setFieldValue("title", new
                StringFieldValue("Britney Jean Spears (born December 2, 1981) is an American singer."));
        doc.setFieldValue("text", new
                StringFieldValue("This is the text "));
        p = getProcessing(new DocumentPut(doc));
        docproc.process(p);
        TensorFieldValue newEmbedding = (TensorFieldValue) doc.getFieldValue("mini_document_embedding");
        assertNotNull(newEmbedding, "Embedding should not be null");
        assertNotEquals(embedding,newEmbedding);

    }

    @Test
    public void test_embedding_already_set() throws Exception {
        Document doc = new Document(createDocType(), "id:msmarco:passage::0");
        doc.setFieldValue("text", new
                StringFieldValue("Britney Jean Spears"));
        TensorType type = new TensorType.Builder(TensorType.Value.FLOAT).
                indexed("d0", 384).build();
        IndexedTensor.Builder builder = IndexedTensor.Builder.of(type);
        for (int i = 0; i < 384; ++i) {
            builder.cell(i,i);
        }
        TensorFieldValue embedding = new TensorFieldValue(builder.build());
        doc.setFieldValue("mini_document_embedding", embedding);
        Processing p = getProcessing(new DocumentPut(doc));
        DocumentProcessor docproc = new DenseDocumentEmbedder(embedder);
        docproc.process(p);
        TensorFieldValue newEmbedding = (TensorFieldValue) doc.getFieldValue("mini_document_embedding");
        assertNotNull(newEmbedding, "Embedding should not be null");
        assertEquals(embedding,newEmbedding, "Embedding should not be touched");
    }

}

