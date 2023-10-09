// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.google.inject.Inject;
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
import com.yahoo.documentapi.DocumentIdResponse;
import com.yahoo.documentapi.DocumentResponse;
import com.yahoo.documentapi.DocumentUpdateResponse;
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
    private static final String VAR_REQ_ID = "reqId";

    // Maps request ID to Document - the lyrics Document looked up
    private final Map<Long, Response> responses = new ConcurrentHashMap<>();

    private final AsyncSession asyncSession;

    class RespHandler implements ResponseHandler {
        @Override
        public void handleResponse(Response response) {
            logger.info("In handleResponse");
            responses.put(response.getRequestId(), response);
        }
    }

    @Inject
    public LyricsDocumentProcessor(DocumentAccess acc) {
        this.asyncSession = acc.createAsyncSession(new AsyncParameters().setResponseHandler(new RespHandler()));
    }

    @Override
    public Progress process(Processing processing) {
        logger.info("In process");
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document document = put.getDocument();
                if (document.getDataType().isA(MUSIC_DOCUMENT_TYPE)) {
                    Long reqId = (Long)processing.getVariable(VAR_REQ_ID);
                    if (reqId != null) { // A request has been made, check for response
                        Response response = responses.get(reqId);
                        if (response != null) {
                            handleResponse(document, reqId, response);
                            responses.remove(reqId);
                            logger.info("  Set lyrics, Progress.DONE");
                            return DocumentProcessor.Progress.DONE;
                        }
                    } else { // Make a lyric request
                        String[] parts = document.getId().toString().split(":");
                        Result res = asyncSession.get(new DocumentId("id:mynamespace:" + LYRICS_DOCUMENT_TYPE + "::" + parts[parts.length-1]));
                        if (res.isSuccess()) {
                            processing.setVariable(VAR_REQ_ID, res.getRequestId());
                            logger.info("  Added to requests pending: " + res.getRequestId());
                        } else {
                            logger.info("  Sending Get failed, Progress.DONE");
                            return DocumentProcessor.Progress.DONE;   // up to the app how to handle such failures, here OK without lyrics
                        }
                    }
                    logger.info("  Request pending ID: " + (long)processing.getVariable(VAR_REQ_ID) + ", Progress.LATER");
                    return DocumentProcessor.Progress.LATER;
                }
            }
        }
        return DocumentProcessor.Progress.DONE;
    }

    private void handleResponse(Document document, long reqId, Response response) {
        if (response.isSuccess()) {
            if (response instanceof DocumentResponse) {
                logger.info("  Async response to put or get, requestID: " + reqId);
                DocumentResponse resp = (DocumentResponse) response;
                setLyrics(document, resp.getDocument());
            } else if (response instanceof DocumentUpdateResponse) {
                logger.info("  Async response to update, requestID: " + reqId);
            } else if (response instanceof DocumentIdResponse) {
                logger.info("  Async response to remove, requestID: " + reqId);
            } else {
                logger.info("  Response, requestID: " + reqId);
            }
        } else {
            logger.info("  Unsuccessful response");
        }
    }

    private void setLyrics(Document document, Document lyricsDoc) {
        if (lyricsDoc != null) {
            FieldValue lyrics = lyricsDoc.getFieldValue("song_lyrics");
            if (lyrics != null) {
                logger.info("  Found lyrics for : " + lyricsDoc.toString());
                document.setFieldValue("lyrics", lyrics);
            }
        } else {
            logger.info("  Get failed, lyrics not found");
        }
    }
    
    @Override
    public void deconstruct() {
        super.deconstruct();
        asyncSession.destroy();
    }

}
