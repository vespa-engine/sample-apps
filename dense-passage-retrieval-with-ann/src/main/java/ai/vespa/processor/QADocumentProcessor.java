// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.processor;


import com.google.inject.Inject;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.FieldValue;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.language.process.Embedder;
import com.yahoo.language.wordpiece.WordPieceEmbedder;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.TensorType;
import java.util.List;

/**
 * Document processor which tokenizes text using the wordpiece tokenizer
 */

public class QADocumentProcessor extends DocumentProcessor {

    private WordPieceEmbedder embedder;
    private static String dimensionName = "d0";
    private static int maxlength = 256;
    private static TensorType titleTensorType = new TensorType.Builder(TensorType.Value.FLOAT).indexed(dimensionName, maxlength).build();
    private static TensorType textTensorType = new TensorType.Builder(TensorType.Value.FLOAT).indexed(dimensionName, maxlength).build();

    @Inject
    public QADocumentProcessor(WordPieceEmbedder embedder) {
        this.embedder = embedder;
    }

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document doc = put.getDocument();
                if (!doc.getDataType().getName().equals("wiki")) {
                    continue;
                }
                doc.setFieldValue("text_token_ids", createTensorField(doc.getFieldValue("text"), titleTensorType));
                doc.setFieldValue("title_token_ids", createTensorField(doc.getFieldValue("title"), textTensorType));
            }
        }
        return Progress.DONE;
    }

    private TensorFieldValue createTensorField(FieldValue field, TensorType type) {
        if (!(field instanceof StringFieldValue))
            throw new IllegalArgumentException("Can only create tensor from string field input");
        StringFieldValue data = (StringFieldValue) field;
        List<Integer> tokens = this.embedder.embed(data.getString(), new Embedder.Context("d"));
        if(tokens.size() > maxlength)
            tokens = tokens.subList(0,maxlength);

        float[] token_ids_float_rep = new float[maxlength];
        for (int i = 0; i < tokens.size(); i++)
            token_ids_float_rep[i] = tokens.get(i).floatValue();
        return new TensorFieldValue(IndexedTensor.Builder.of(type, token_ids_float_rep).build());
    }
}
