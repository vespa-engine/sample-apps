//Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.docproc;

import ai.vespa.examples.TokenizerFactory;
import ai.vespa.examples.colbert.ColbertConfig;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.docproc.Processing;
import com.yahoo.document.*;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.tensor.*;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;
import java.util.Optional;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertNotNull;


public class ColBERTDocumentProcessorTest {

    static ModelsEvaluator evaluator = ModelsEvaluatorTester.create("src/main/application/models/");


    private static ColbertConfig getColBertConfig() {
        ColbertConfig.Builder builder = new ColbertConfig.Builder();
        return builder.max_document_length(64).max_query_length(32).dim(32).build();
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
        type.addField("title", DataType.STRING);
        type.addField("text", DataType.STRING);
        type.addField("dt",
                TensorDataType.getTensor(TensorType.fromSpec("tensor<bfloat16>(dt{},x[32])")));
        return type;
    }

    @Test
    public void test_happy_path() throws Exception {
        Document doc = new Document(createDocType(), "id:msmarco:passage::3");
        doc.setFieldValue("text", new
                StringFieldValue("The Manhattan Project was the name for a project conducted during World War II," +
                " to develop the first atomic bomb. It refers specifically to the period of the project " +
                "from 194 \u2026 2-1946 under the control of the U.S. Army Corps of Engineers, " +
                "under the administration of General Leslie R. Groves."));
        ColBERTDocumentProcessor processor = new ColBERTDocumentProcessor(evaluator, TokenizerFactory.getEmbedder(),getColBertConfig());
        Processing processing = getProcessing(new DocumentPut(doc));
        processor.process(processing);
        TensorFieldValue dt_tensor = (TensorFieldValue) doc.getFieldValue("dt");
        assertNotNull(dt_tensor);
        Optional<Tensor> t = dt_tensor.getTensor();
        assertTrue(t.isPresent());
        MixedTensor m = (MixedTensor)t.get();
        assertEquals(32, m.denseSubspaceSize());
    }
}
