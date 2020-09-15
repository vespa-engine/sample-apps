package ai.vespa.processor;

import ai.vespa.tokenizer.BertTokenizer;
import com.google.inject.Inject;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.FieldValue;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.tensor.Tensor;

import java.util.List;


public class QADocumentProcessor extends DocumentProcessor {

    BertTokenizer tokenizer;

    @Inject
    public QADocumentProcessor(BertTokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document doc = put.getDocument();
                doc.setFieldValue("text_token_ids", createTensorField(doc.getFieldValue("text")));
                doc.setFieldValue("title_token_ids", createTensorField(doc.getFieldValue("title")));
            }
        }
        return Progress.DONE;
    }

    private TensorFieldValue createTensorField(FieldValue field) {
        StringFieldValue data = (StringFieldValue)field;
        List<Integer> token_ids = this.tokenizer.tokenize(data.getString(),true);
        String tensorSpec = "tensor<float>(d0[" + tokenizer.getMaxLength() + "]):" + token_ids ;
        return new TensorFieldValue(Tensor.from(tensorSpec));
    }
}
