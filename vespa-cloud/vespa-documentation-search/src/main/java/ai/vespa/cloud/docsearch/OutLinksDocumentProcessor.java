// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.cloud.docsearch;

import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentType;
import com.yahoo.document.DocumentUpdate;
import com.yahoo.document.datatypes.Array;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.WeightedSet;
import com.yahoo.document.update.FieldUpdate;
import com.yahoo.documentapi.AsyncParameters;
import com.yahoo.documentapi.AsyncSession;
import com.yahoo.documentapi.Response;
import com.yahoo.documentapi.ResponseHandler;
import com.yahoo.documentapi.Result;
import com.yahoo.documentapi.messagebus.MessageBusDocumentAccess;

import java.nio.file.Path;
import java.util.Collections;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Extract unique file-links from this document, and add as an in-link in the target document.
 * Use async Document API to set in-links, fire-and-forget.
 * This is an example Document Processor / use of Document API - and to be tested in a SystemTest.
 */
public class OutLinksDocumentProcessor extends DocumentProcessor {
    private static final Logger logger = Logger.getLogger(OutLinksDocumentProcessor.class.getName());

    static final String DOC_DOCUMENT_TYPE   = "doc";
    static final String PATH_FIELD_NAME     = "path";
    static final String OUTLINKS_FIELD_NAME = "outlinks";
    static final String INLINKS_FIELD_NAME  = "inlinks";

    private final MessageBusDocumentAccess access = new MessageBusDocumentAccess();
    private final AsyncSession asyncSession = access.createAsyncSession(new AsyncParameters().setResponseHandler(new RespHandler()));

    class RespHandler implements ResponseHandler {
        @Override
        public void handleResponse(Response response) {
            //logger.info("In handleResponse");
        }
    }

    @Override
    public Progress process(Processing processing) {
        //logger.info("In process");
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document document = put.getDocument();
                if (document.getDataType().isA(DOC_DOCUMENT_TYPE)) {
                    String myPath = document.getFieldValue(PATH_FIELD_NAME).toString();

                    Set<String> docsLinkedFromThis = canonicalizeLinks(Path.of(myPath),
                            onlyFileLinks(removeLinkFragment(document)));

                    Array<StringFieldValue> sfv = new Array<>(DataType.getArray(DataType.STRING));
                    for (String link : docsLinkedFromThis) {
                        sfv.add(new StringFieldValue(link));
                    }
                    document.setFieldValue(OUTLINKS_FIELD_NAME, sfv);

                    if (docsLinkedFromThis.size() > 0) {
                        addInLinks(docsLinkedFromThis, myPath, document.getDataType());
                    }
                }
            }
        }
        return Progress.DONE;
    }

    @SuppressWarnings("unchecked")
    private void addInLinks(Set<String> targetDocs, String myPath, DocumentType docType) {
        WeightedSet<StringFieldValue> wset = new WeightedSet<>(DataType.getWeightedSet(DataType.STRING));
        wset.put(new StringFieldValue(myPath), 1);
        for (String targetDoc : targetDocs) {
            String targetID = "id:open:" + DOC_DOCUMENT_TYPE + "::open" + targetDoc;
            DocumentUpdate upd = new DocumentUpdate(docType, targetID);
            upd.addFieldUpdate(FieldUpdate.createAddAll(docType.getField(INLINKS_FIELD_NAME), wset));
            Result res = asyncSession.update(upd);
            if (res.isSuccess()) {
                //logger.info("Added to requests pending: " + res.getRequestId());
            } else {
                //logger.info("Sending Update failed");
            }
        }
    }

    protected static Set<String> onlyFileLinks(Set<String> links) {
        return links.stream()
                .filter(l -> !(l.startsWith("http://")))
                .filter(l -> !(l.startsWith("https://")))
                .collect(Collectors.toSet());
    }

    protected static Set<String> canonicalizeLinks(Path docPath, Set<String> links) {
        // Rewrite links to absolute path
        // Path is like /documentation/operations/admin-procedures.html
        return links.stream()
                .map(l -> docPath.getParent().resolve(l).normalize().toString())
                .collect(Collectors.toSet());
    }

    @SuppressWarnings("unchecked")
    protected static Set<String> removeLinkFragment(Document doc) {
        // Remove fragment after # like vespa-cmdline-tools.html#vespa-model-inspect
        Array<StringFieldValue> outLinks = (Array<StringFieldValue>) doc.getFieldValue(OUTLINKS_FIELD_NAME);
        if (outLinks == null) { return Collections.emptySet(); }
        return outLinks.stream()
                .map(l -> l.toString().replaceFirst("#.*$", ""))
                .filter(l -> !(l.isEmpty()))
                .collect(Collectors.toSet());
    }
}
