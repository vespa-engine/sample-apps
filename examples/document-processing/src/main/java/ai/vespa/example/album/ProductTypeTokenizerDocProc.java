// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.Array;
import com.yahoo.document.datatypes.StringFieldValue;

import java.util.logging.Logger;

/**
* Expand a string to an array of substrings
* From:
*     "producttype": "Media > Music & Sound Recordings > Music Cassette Tapes"
* To:
*     "producttypetokens": [
*     "",
*     "M",
*     "Me",
*     "Med",
*     "Medi",
*     "Media",
*     "Media ",
*     "Media >",
*     "Media > ",
*     "Media > M",
* ...
*     "Media > Music & Sound Recordings > Music Cassette Tape",
*     "Media > Music & Sound Recordings > Music Cassette Tapes"
*    ],
*
* This to be used in exact matching of the substrings, example query:
* curl "$ENDPOINT/search/?yql=select%20%2A%20from%20sources%20%2A%20where%20producttypetokens%20contains%20%22Media%20%3E%20Music%20%26%20Sou%22"
*
* To deploy, modify services.xml to:
* <documentprocessor id="ai.vespa.example.album.ProductTypeTokenizerDocProc" bundle="albums-docproc"/>
*/
public class ProductTypeTokenizerDocProc extends DocumentProcessor {
    private static final Logger logger = Logger.getLogger(ProductTypeTokenizerDocProc.class.getName());
    protected static final String MUSIC_DOCUMENT_TYPE            = "music";
    protected static final String PRODUCT_TYPE_FIELD_NAME        = "producttype";
    protected static final String PRODUCT_TYPE_TOKENS_FIELD_NAME = "producttypetokens";

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
        Array<StringFieldValue> tokens = new Array<>(DataType.getArray(DataType.STRING));
        String productType = document.getFieldValue(PRODUCT_TYPE_FIELD_NAME).toString();
        int len = productType.length();
        for (int i=1; i<=len; i++) {
            tokens.add(new StringFieldValue(productType.substring(0,i)));
        }
        document.setFieldValue(PRODUCT_TYPE_TOKENS_FIELD_NAME, tokens);
    }

    @Override
    public void deconstruct() {
        super.deconstruct();  // no need for other deconstruction here
    }
}
