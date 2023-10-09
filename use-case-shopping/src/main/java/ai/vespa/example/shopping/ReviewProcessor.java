// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping;

import com.google.inject.Inject;
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
import com.yahoo.documentapi.AsyncParameters;
import com.yahoo.documentapi.AsyncSession;
import com.yahoo.documentapi.DocumentAccess;
import com.yahoo.documentapi.DocumentIdResponse;
import com.yahoo.documentapi.DocumentResponse;
import com.yahoo.documentapi.DocumentUpdateResponse;
import com.yahoo.documentapi.Response;
import com.yahoo.documentapi.ResponseHandler;
import com.yahoo.documentapi.Result;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

public class ReviewProcessor extends DocumentProcessor {

    private static Logger log = Logger.getLogger(ReviewProcessor.class.getName());

    private final static String REVIEW_DOCUMENT_TYPE = "review";
    private final static String ITEM_DOCUMENT_TYPE = "item";
    private final static String ASIN_FIELD_NAME = "asin";
    private final static String STARS_FIELD_NAME = "stars";
    private final static String RATING_STARS_FIELD_NAME = "rating_stars";
    private final static String RATING_COUNT_FIELD_NAME = "rating_count";

    private final DocumentAccess access;
    private final AsyncSession asyncSession;
    private static final String VAR_REQ_ID = "reqId";

    private final Map<Long, Response> responses = new ConcurrentHashMap<>();

    @Inject
    public ReviewProcessor(DocumentAccess documentAccess) {
        access = documentAccess;
        asyncSession = access.createAsyncSession(new AsyncParameters().setResponseHandler(new RespHandler()));
    }

    class RespHandler implements ResponseHandler {
        @Override
        public void handleResponse(Response response) {
            responses.put(response.getRequestId(), response);
        }
    }

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document document = put.getDocument();
                if (document.getDataType().isA(REVIEW_DOCUMENT_TYPE)) {
                    Long reqId = (Long)processing.getVariable(VAR_REQ_ID);
                    if (reqId != null) { // An update has been made, check for response
                        Response response = responses.get(reqId);
                        if (response != null) {
                            handleResponse(document, reqId, response);
                            responses.remove(reqId);
                            return Progress.DONE;
                        }
                    } else { // Update rating
                        StringFieldValue asin = (StringFieldValue) document.getFieldValue(ASIN_FIELD_NAME);
                        IntegerFieldValue stars = (IntegerFieldValue) document.getFieldValue(STARS_FIELD_NAME);
                        if (asin != null && stars != null) {
                            DocumentId id = new DocumentId("id:" + ITEM_DOCUMENT_TYPE + ":" + ITEM_DOCUMENT_TYPE + "::" + asin.getString());
                            DocumentType type = access.getDocumentTypeManager().getDocumentType(ITEM_DOCUMENT_TYPE);
                            DocumentUpdate update = new DocumentUpdate(type, id);
                            update.addFieldUpdate(FieldUpdate.createIncrement(type.getField(RATING_STARS_FIELD_NAME), stars.getNumber()));
                            update.addFieldUpdate(FieldUpdate.createIncrement(type.getField(RATING_COUNT_FIELD_NAME), 1));
                            Result res = asyncSession.update(update);
                            if (res.isSuccess()) {
                                processing.setVariable(VAR_REQ_ID, res.getRequestId());
                            } else {
                                // Can happen if review is fed for a non-existing item. Skip for now.
                                return Progress.DONE;
                            }
                        }

                    }
                    return Progress.LATER;
                }
            }
        }
        return Progress.DONE;
    }

    private void handleResponse(Document document, long reqId, Response response) {
        if (response.isSuccess()) {
            if (response instanceof DocumentResponse) {
            } else if (response instanceof DocumentUpdateResponse) { // do nothing ...
            } else if (response instanceof DocumentIdResponse) {
            } else { }
        } else { } //  Unsuccessful response
    }

    @Override
    public void deconstruct() {
        super.deconstruct();
        asyncSession.destroy();
    }

}
