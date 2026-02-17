package ai.vespa.examples.docproc;

import ai.vespa.examples.DimensionReducer;
import com.yahoo.component.chain.dependencies.Provides;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;

@Provides("DimensionReduction")
public class DimensionReductionDocProc extends DocumentProcessor {

    private final DimensionReducer reducer;

    public DimensionReductionDocProc(DimensionReducer reducer) {
        this.reducer = reducer;
    }

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut put) {
                Document doc = put.getDocument();
                TensorFieldValue vector = (TensorFieldValue) doc.getFieldValue("vector");
                if (vector == null || vector.getTensor().isEmpty()) {
                    return Progress.FAILED.withReason("No 'vector' tensor field in document");
                }
                Tensor fullVector =  vector.getTensor().get();
                Tensor reducedVector = reducer.reduce(fullVector);
                TensorFieldValue tensorFieldValue = new TensorFieldValue(
                        reducedVector.cellCast(TensorType.Value.BFLOAT16));
                doc.setFieldValue("reduced_vector", tensorFieldValue);
                if(doc.getDataType().getName().equals("centroid"))
                    doc.removeFieldValue("vector");
            }
        }
        return Progress.DONE;
    }
}
