// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.processor;

import com.yahoo.config.FileReference;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentType;
import com.yahoo.document.TensorDataType;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.language.wordpiece.WordPieceConfig;
import com.yahoo.language.wordpiece.WordPieceEmbedder;
import com.yahoo.tensor.TensorType;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class QADocumentProcessorTest {

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

    private static DocumentType createWikiDocType() {
        DocumentType type = new DocumentType("wiki");
        type.addField("title", DataType.STRING);
        type.addField("title_token_ids", TensorDataType.getTensor(TensorType.fromSpec("tensor<float>(d0[256])")));
        type.addField("text", DataType.STRING);
        type.addField("text_token_ids", TensorDataType.getTensor(TensorType.fromSpec("tensor<float>(d0[256])")));
        return type;
    }

    @Test
    public void testProcessing() throws Exception {
        Document doc = new Document(createWikiDocType(), "id:foo:wiki::0");
        doc.setFieldValue("title", new StringFieldValue("Britney_spears"));
        doc.setFieldValue("text", new StringFieldValue("Britney Jean Spears (born December 2, 1981) is an American singer, songwriter, dancer, and actress."));
        Processing p = getProcessing(new DocumentPut(doc));
        QADocumentProcessor processor = new QADocumentProcessor(embedder);
        processor.process(p);
        TensorFieldValue title_tensor = (TensorFieldValue)doc.getFieldValue("title_token_ids");
        TensorFieldValue text_tensor = (TensorFieldValue)doc.getFieldValue("text_token_ids");
        assertNotNull(title_tensor);
        assertNotNull(text_tensor);
        assertTrue(title_tensor.toString().startsWith("tensor<float>(d0[256]):[29168.0, 1035.0, 13957.0"));
        assertTrue(text_tensor.toString().startsWith("tensor<float>(d0[256]):[29168.0, 3744.0, 13957.0, 1006.0, 2141.0, 2285.0, 1016.0, 1010.0"));
    }

    @Test
    public void testEmptyProcessing() throws Exception  {
        Document doc = new Document(new DocumentType("query"), "id:foo:query::0");
        Processing p = getProcessing(new DocumentPut(doc));
        QADocumentProcessor processor = new QADocumentProcessor(embedder);
        processor.process(p);
    }
}
