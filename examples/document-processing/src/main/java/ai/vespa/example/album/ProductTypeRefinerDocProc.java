// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;

import java.util.logging.Logger;

public class ProductTypeRefinerDocProc extends DocumentProcessor {
    private static final Logger logger = Logger.getLogger(ProductTypeRefinerDocProc.class.getName());
    protected static final String MUSIC_DOCUMENT_TYPE     = "music";
    protected static final String PRODUCT_TYPE_FIELD_NAME = "producttype";

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document document = put.getDocument();
                if (document.getDataType().isA(MUSIC_DOCUMENT_TYPE)) {
                    logger.info("Before processing: " + document.toJson());
                    processPut(document);
                    logger.info("After processing:  " + document.toJson());
                    return Progress.DONE;
                }
            }
        }
        return Progress.DONE;
    }

    private void processPut(Document document) {
        document.setFieldValue(PRODUCT_TYPE_FIELD_NAME,
                document.getFieldValue(PRODUCT_TYPE_FIELD_NAME).toString().replace('>','|'));
    }

}
