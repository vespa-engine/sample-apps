// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.docproc;

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
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.TensorType;

import java.util.List;

public class DocumentTensorizer extends DocumentProcessor {

    private BertTokenizer tokenizer;
    private static TensorType textTensorType = new TensorType.Builder(TensorType.Value.FLOAT).indexed("d0", 128).build();

    @Inject
    public DocumentTensorizer(BertTokenizer tokenizer) {
        this.tokenizer = tokenizer;
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
                if (doc.getFieldValue("text") != null)
                    doc.setFieldValue("text_token_ids", createTensorField(doc.getFieldValue("text"), textTensorType));
            }
        }
        return Progress.DONE;
    }

    private TensorFieldValue createTensorField(FieldValue field, TensorType type) {
        if (!(field instanceof StringFieldValue))
            throw new IllegalArgumentException("Can only create tensor from string field input");
        StringFieldValue data = (StringFieldValue) field;
        int maxLength = type.sizeOfDimension("d0").get().intValue();
        List<Integer> token_ids = this.tokenizer.tokenize(data.getString(), maxLength, true);
        float[] token_ids_float_rep = new float[token_ids.size()];
        for (int i = 0; i < token_ids.size(); i++)
            token_ids_float_rep[i] = token_ids.get(i).floatValue();
        return new TensorFieldValue(IndexedTensor.Builder.of(type, token_ids_float_rep).build());
    }
}
