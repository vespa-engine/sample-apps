package ai.vespa.examples.docproc;

import ai.vespa.examples.DimensionReducer;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.docproc.Processing;
import com.yahoo.document.*;
import com.yahoo.document.datatypes.IntegerFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.TensorType;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;


public class DocumentReductionDocProcTest {

    static DimensionReductionDocProc processor;
    static {
        ModelsEvaluator evaluator = ModelsEvaluatorTester.create("src/main/application/models");
        DimensionReducer reducer = new DimensionReducer(evaluator);
        processor = new DimensionReductionDocProc(reducer);
    }

    private static Processing getProcessing(DocumentOperation... operations) {
        Processing processing = new Processing();
        for (DocumentOperation op : operations) {
            processing.addDocumentOperation(op);
        }
        return processing;
    }

    private static DocumentType createType(String name) {
        DocumentType type = new DocumentType(name);
        type.addField("id", DataType.STRING);
        type.addField("vector",
                TensorDataType.getTensor(TensorType.fromSpec("tensor<bfloat16>(x[768])")));
        type.addField("reduced_vector",
                TensorDataType.getTensor(TensorType.fromSpec("tensor<bfloat16>(x[128])")));
        return type;
    }

    @Test
    public void test_happy_path()  {
        Document centroid = new Document(createType("centroid"), "id:laion:centroid::3456781");
        centroid.setFieldValue("id", new IntegerFieldValue(3456781));
        centroid.setFieldValue("vector", getTensor());

        Document image = new Document(createType("image"), "id:laion:image::904384");
        image.setFieldValue("id", new IntegerFieldValue(904384));
        image.setFieldValue("vector", getTensor());
        Processing processing = getProcessing(new DocumentPut(centroid), new DocumentPut(image));
        processor.process(processing);

        TensorFieldValue vectorReduced = (TensorFieldValue)centroid.getFieldValue("reduced_vector");
        assertTrue(vectorReduced.getTensor().isPresent());
        assertEquals("tensor<bfloat16>(x[128])",
                vectorReduced.getTensor().get().toAbbreviatedString().split(":")[0]);
        assertNull(centroid.getFieldValue("vector"));

        vectorReduced = (TensorFieldValue)image.getFieldValue("reduced_vector");
        assertTrue(vectorReduced.getTensor().isPresent());
        assertEquals("tensor<bfloat16>(x[128])",
                vectorReduced.getTensor().get().toAbbreviatedString().split(":")[0]);

        assertTrue(((TensorFieldValue)image.getFieldValue("vector")).getTensor().isPresent());
    }

    static TensorFieldValue getTensor() {
        TensorType type = new TensorType.Builder(TensorType.Value.BFLOAT16).indexed("x", 768).build();
        IndexedTensor.Builder builder = IndexedTensor.Builder.of(type);
        for (int j = 0; j < 768; j++)
            builder.cell(1.0, j);
        return new TensorFieldValue(builder.build());
    }
}
