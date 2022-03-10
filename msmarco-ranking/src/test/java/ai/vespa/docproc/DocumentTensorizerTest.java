// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.docproc;

import com.yahoo.config.FileReference;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.*;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.language.wordpiece.WordPieceConfig;
import com.yahoo.language.wordpiece.WordPieceEmbedder;
import com.yahoo.tensor.TensorType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;

public class DocumentTensorizerTest {

    private static WordPieceEmbedder getEmbedder() {
        WordPieceConfig.Builder b = new WordPieceConfig.Builder().model(
                new WordPieceConfig.Model.Builder()
                        .language("unknown")
                        .path(new FileReference("src/main/application/files/bert-base-uncased-vocab.txt")));
        return new WordPieceEmbedder(b.build());
    }

    private static WordPieceEmbedder embedder = getEmbedder();

    private static Processing getProcessing(DocumentOperation... operations) {
        Processing processing = new Processing();
        for (DocumentOperation op : operations) {
            processing.addDocumentOperation(op);
        }
        return processing;
    }

    private static DocumentType createDocType() {
        DocumentType type = new DocumentType("passage");
        type.addField("text", DataType.STRING);
        type.addField("text_token_ids", TensorDataType.getTensor(TensorType.fromSpec("tensor<float>(d0[128])")));
        return type;
    }

    @Test
    public void testProcessing() throws Exception {
        Document doc = new Document(createDocType(), "id:msmarco:passage::0");
        doc.setFieldValue("text", new
                StringFieldValue("Britney Jean Spears (born December 2, 1981) is an American singer, songwriter, dancer, and actress."));
        DocumentProcessor processor = new DocumentTensorizer(embedder);
        processor.process(getProcessing(new DocumentPut(doc)));
        TensorFieldValue text_tensor = (TensorFieldValue) doc.getFieldValue("text_token_ids");
        assertNotNull(text_tensor);
    }
}
