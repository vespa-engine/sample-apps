// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.fasterxml.jackson.core.JsonFactory;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentTypeManager;
import com.yahoo.document.json.DocumentOperationType;
import com.yahoo.document.json.JsonReader;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.logging.Logger;

import static ai.vespa.example.album.ProductTypeRefinerDocProc.PRODUCT_TYPE_FIELD_NAME;

/**
 * The DocTypeDocumentProcessor demonstrates how to instantiate a Document from JSON
 * It also illustrates how to get/use DocumentType(Manager) in code and tests, see DocTypeDocProcTest
 */
public class DocTypeDocumentProcessor extends DocumentProcessor {
    private static final Logger logger = Logger.getLogger(DocTypeDocumentProcessor.class.getName());
    protected static final String MUSIC_DOCUMENT_TYPE = "music";
    protected static final String ARTIST_FIELD_NAME   = "artist";

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut)op;
                Document document = put.getDocument();
                if (document.getDataType().isA(MUSIC_DOCUMENT_TYPE)) {
                    logger.info("Before processing: " + document.toJson());

                    Document documentFromJson = createDocFromJson(processing.getService().getDocumentTypeManager(),
                            "id:mynamespace:" + MUSIC_DOCUMENT_TYPE + "::2").getDocument();
                    document.setFieldValue(ARTIST_FIELD_NAME, documentFromJson.getFieldValue(ARTIST_FIELD_NAME));

                    logger.info("After processing:  " + document.toJson());
                    return Progress.DONE;
                }
            }
        }
        return Progress.DONE;
    }

    public DocumentPut createDocFromJson(DocumentTypeManager typeMgr, String docId) {
        String docFields = "{ \"fields\": {\"" + ARTIST_FIELD_NAME + "\": \"Elvis\"} }";
        JsonReader reader = new JsonReader(
                typeMgr,
                new ByteArrayInputStream(docFields.getBytes(StandardCharsets.UTF_8)),
                new JsonFactory());
        return (DocumentPut)reader.readSingleDocument(DocumentOperationType.PUT, docId);
    }
}
