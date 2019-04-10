// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping;

import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentId;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentType;
import com.yahoo.document.DocumentUpdate;
import com.yahoo.document.datatypes.IntegerFieldValue;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.update.FieldUpdate;
import com.yahoo.documentapi.DocumentAccess;
import com.yahoo.documentapi.SyncParameters;
import com.yahoo.documentapi.SyncSession;

import java.util.logging.Logger;

public class ReviewProcessor extends DocumentProcessor {

    private static Logger log = Logger.getLogger(ReviewProcessor.class.getName());

    private final static String REVIEW_DOCUMENT_TYPE = "review";
    private final static String ITEM_DOCUMENT_TYPE = "item";
    private final static String ASIN_FIELD_NAME = "asin";
    private final static String STARS_FIELD_NAME = "stars";
    private final static String RATING_STARS_FIELD_NAME = "rating_stars";
    private final static String RATING_COUNT_FIELD_NAME = "rating_count";

    private final DocumentAccess access = DocumentAccess.createDefault();
    private final SyncSession session = access.createSyncSession(new SyncParameters.Builder().build());

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document document = put.getDocument();
                if (document.getDataType().isA(REVIEW_DOCUMENT_TYPE)) {
                    updateItemRating(document);
                }
            }

        }
        return Progress.DONE;
    }

    private void updateItemRating(Document document) {
        StringFieldValue asin = (StringFieldValue) document.getFieldValue(ASIN_FIELD_NAME);
        IntegerFieldValue stars = (IntegerFieldValue) document.getFieldValue(STARS_FIELD_NAME);
        if (asin != null && stars != null) {
            DocumentId id = new DocumentId("id:" + ITEM_DOCUMENT_TYPE + ":" + ITEM_DOCUMENT_TYPE + "::" + asin.getString());
            DocumentType type = access.getDocumentTypeManager().getDocumentType(ITEM_DOCUMENT_TYPE);
            DocumentUpdate update = new DocumentUpdate(type, id);
            update.addFieldUpdate(FieldUpdate.createIncrement(type.getField(RATING_STARS_FIELD_NAME), stars.getNumber()));
            update.addFieldUpdate(FieldUpdate.createIncrement(type.getField(RATING_COUNT_FIELD_NAME), 1));
            if ( ! session.update(update)) {
                // Can happen if review is fed for a non-existing item. Skip for now.
            }
        }
    }
}
