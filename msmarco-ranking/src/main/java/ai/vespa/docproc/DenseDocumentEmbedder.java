package ai.vespa.docproc;

import ai.vespa.embedding.DenseEmbedder;
import com.google.inject.Inject;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.FieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.tensor.Tensor;
import java.util.Optional;


public class DenseDocumentEmbedder extends DocumentProcessor {

    private final DenseEmbedder denseEmbedder;

    @Inject
    public DenseDocumentEmbedder(DenseEmbedder denseEmbedder) {
        this.denseEmbedder = denseEmbedder;
    }

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document doc = put.getDocument();
                String type = doc.getDataType().getName();
                if (!(type.equals("passage"))) {
                    continue;
                }
                FieldValue e = doc.getFieldValue("mini_document_embedding");
                FieldValue text = doc.getFieldValue("text");
                FieldValue title = doc.getFieldValue("title");
                //only embed if there is not already an embedding
                if(e != null)
                    continue;
                if (text != null && text.getDataType() != DataType.STRING)
                    return Progress.FAILED.withReason(("Input field 'text' is not string field"));

                if (title != null && title.getDataType() != DataType.STRING)
                    return Progress.FAILED.withReason(("Input field 'title' is not string field"));

                Optional<Tensor> optionalEmbedding = this.denseEmbedder.embed(
                        doc,"title","text");
                if(optionalEmbedding.isEmpty())
                    return Progress.FAILED.withReason("Unable to compute embedding for document");
                TensorFieldValue embedding = new TensorFieldValue(optionalEmbedding.get());
                doc.setFieldValue("mini_document_embedding",embedding);
            }
        }
        return Progress.DONE;
    }
}
