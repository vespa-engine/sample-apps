// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentId;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.FieldValue;
import com.yahoo.documentapi.AsyncParameters;
import com.yahoo.documentapi.AsyncSession;
import com.yahoo.documentapi.DocumentAccess;
import com.yahoo.documentapi.DocumentResponse;
import com.yahoo.documentapi.Response;
import com.yahoo.documentapi.ResponseHandler;
import com.yahoo.documentapi.Result;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

public class LyricsDocumentProcessor extends DocumentProcessor {
    private static final Logger logger = Logger.getLogger(LyricsDocumentProcessor.class.getName());

    private static final String MUSIC_DOCUMENT_TYPE  = "music";
    private static final String LYRICS_DOCUMENT_TYPE = "lyrics";
    private static final String VAR_LYRICS = "lyrics";
    private static final String VAR_REQ_ID = "reqId";

    // Maps request IDs to Processing instances
    private final Map<Long, Processing> processings = new ConcurrentHashMap<>();

    private final DocumentAccess access     = DocumentAccess.createDefault();
    private final AsyncSession asyncSession = access.createAsyncSession(new AsyncParameters().setResponseHandler(new RespHandler()));

    class RespHandler implements ResponseHandler {
        @Override
        public void handleResponse(Response response) {
            logger.info("In handleResponse");
            if (response.isSuccess()){
                long reqId = response.getRequestId();
                logger.info("  requestID: " + reqId);
                if (response instanceof DocumentResponse) {
                    Processing processing = processings.remove(reqId);
                    processing.removeVariable(VAR_REQ_ID);
                    DocumentResponse resp = (DocumentResponse)response;
                    Document doc = resp.getDocument();
                    if (doc != null) {
                        logger.info("  Found lyrics for : " + doc.toString());
                        processing.setVariable(VAR_LYRICS, resp.getDocument().getFieldValue("song_lyrics"));
                    }
                }
            }
        }
    }

    @Override
    public Progress process(Processing processing) {
        logger.info("In process");
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document document = put.getDocument();
                if (document.getDataType().isA(MUSIC_DOCUMENT_TYPE)) {

                    // Set lyrics if already looked up, and set DONE
                    FieldValue lyrics = (FieldValue)processing.getVariable(VAR_LYRICS);
                    if (lyrics != null) {
                        document.setFieldValue("lyrics", lyrics);
                        logger.info("  Set lyrics, Progress.DONE");
                        return DocumentProcessor.Progress.DONE;
                    }

                    // Make a lyric request if not already pending
                    if (processing.getVariable(VAR_REQ_ID) == null) {
                        String[] parts = document.getId().toString().split(":");
                        Result res = asyncSession.get(new DocumentId("id:mynamespace:" + LYRICS_DOCUMENT_TYPE + "::" + parts[parts.length-1]));
                        if (res.isSuccess()) {
                            processing.setVariable(VAR_REQ_ID, res.getRequestId());
                            processings.put(res.getRequestId(), processing);
                            logger.info("  Added to requests pending: " + res.getRequestId());
                        }
                    }
                    logger.info("  Request pending ID: " + (long)processing.getVariable(VAR_REQ_ID) + ", Progress.LATER");
                    return DocumentProcessor.Progress.LATER;
                }
            }
        }
        return DocumentProcessor.Progress.DONE;
    }

    @Override
    public void deconstruct() {
        super.deconstruct();
        asyncSession.destroy();
        access.shutdown();
    }
}
