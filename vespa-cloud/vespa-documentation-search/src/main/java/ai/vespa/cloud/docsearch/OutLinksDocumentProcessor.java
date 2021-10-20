// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.cloud.docsearch;

import com.google.inject.Inject;
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
import com.yahoo.document.update.ValueUpdate;
import com.yahoo.documentapi.AsyncParameters;
import com.yahoo.documentapi.AsyncSession;
import com.yahoo.documentapi.DocumentAccess;
import com.yahoo.documentapi.Response;
import com.yahoo.documentapi.ResponseHandler;
import com.yahoo.documentapi.Result;

import java.nio.file.Path;
import java.util.Collections;
import java.util.HashSet;
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

    static final String DOC_DOCUMENT_TYPE    = "doc";
    static final String PATH_FIELD_NAME      = "path";
    static final String NAMESPACE_FIELD_NAME = "namespace";
    static final String TITLE_FIELD_NAME     = "title";
    static final String CONTENT_FIELD_NAME   = "content";
    static final String OUTLINKS_FIELD_NAME  = "outlinks";
    static final String INLINKS_FIELD_NAME   = "inlinks";

    private final AsyncSession asyncSession;

    class RespHandler implements ResponseHandler {
        @Override
        public void handleResponse(Response response) {
            /*
            if (response.isSuccess()) {
                logger.info("Success!");
            }
            logger.info("In handleResponse: " + response.getTextMessage());
             */
        }
    }

    @Inject
    public OutLinksDocumentProcessor(DocumentAccess acc) {
        this.asyncSession = acc.createAsyncSession(new AsyncParameters().setResponseHandler(new RespHandler()));
    }

    @Override
    @SuppressWarnings("unchecked")
    public Progress process(Processing processing) {
        //logger.info("In process");
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document document = put.getDocument();
                if (document.getDataType().isA(DOC_DOCUMENT_TYPE)) {
                    String myPath = document.getFieldValue(PATH_FIELD_NAME).toString();

                    Set<String> docsLinkedFromMe = canonicalizeLinks(Path.of(myPath),
                            onlyFileLinks(removeLinkFragment(getUniqueOutLinks(document))));

                    Array<StringFieldValue> sanitizedLinks = new Array<>(DataType.getArray(DataType.STRING));
                    for (String link : docsLinkedFromMe) {
                        sanitizedLinks.add(new StringFieldValue(link));
                    }

                    document.setFieldValue(OUTLINKS_FIELD_NAME, sanitizedLinks);

                    if (docsLinkedFromMe.size() > 0) {
                        addInLinkToOtherDocs(docsLinkedFromMe, myPath, document.getDataType());
                    }
                }
            }
            else if (op instanceof DocumentUpdate) {
                DocumentUpdate update = (DocumentUpdate) op;
                if (update.getType().isA(DOC_DOCUMENT_TYPE)) {
                     /*
                      There are two kinds of updates:
                      * a create-if-nonexistent update for a new document - if it has a PATH_FIELD_NAME
                      * updating fields in documents

                      The first needs the processing below - the second does not need processing, just send the update as-is
                     */

                    FieldUpdate fieldUpdate = update.getFieldUpdate(PATH_FIELD_NAME);
                    if (fieldUpdate == null) { return Progress.DONE;}  // no extra processing

                    ValueUpdate<StringFieldValue> valueUpdate = fieldUpdate.getValueUpdate(0);
                    String myPath = valueUpdate.getValue().getString();

                    Set<String> docsLinkedFromMe = canonicalizeLinks(Path.of(myPath),
                            onlyFileLinks(removeLinkFragment(getUniqueOutLinks(update))));

                    Array<StringFieldValue> sanitizedLinks = new Array<>(DataType.getArray(DataType.STRING));
                    for (String link : docsLinkedFromMe) {
                        sanitizedLinks.add(new StringFieldValue(link));
                    }

                    update.removeFieldUpdate(OUTLINKS_FIELD_NAME); // remove the update with un-sanitized links
                    FieldUpdate sanitizedLinksUpdate = FieldUpdate.createAssign(update.getDocumentType().getField(OUTLINKS_FIELD_NAME), sanitizedLinks);
                    update.addFieldUpdate(sanitizedLinksUpdate);   // ... and add an update with the clean links instead

                    if (docsLinkedFromMe.size() > 0) {
                        addInLinkToOtherDocs(docsLinkedFromMe, myPath, update.getDocumentType());
                    }
                }
            }
        }
        return Progress.DONE;
    }

    @SuppressWarnings("unchecked")
    private void addInLinkToOtherDocs(Set<String> targetDocs, String myPath, DocumentType docType) {
        WeightedSet<StringFieldValue> wset = new WeightedSet<>(DataType.getWeightedSet(DataType.STRING));
        wset.put(new StringFieldValue(myPath), 1);
        for (String targetDoc : targetDocs) {
            String targetID = "id:open:" + DOC_DOCUMENT_TYPE + "::open" + targetDoc;
            //logger.info("Update target doc: " + targetID);
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
    protected static Set<String> getUniqueOutLinks(Document doc) {
        Array<StringFieldValue> outLinks = (Array<StringFieldValue>) doc.getFieldValue(OUTLINKS_FIELD_NAME);
        if (outLinks == null) { return Collections.emptySet(); }
        Set<String> links = new HashSet<>();
        for (StringFieldValue link : outLinks) {
            links.add(link.toString());
        }
        return links;
    }

    @SuppressWarnings("unchecked")
    protected static Set<String> getUniqueOutLinks(DocumentUpdate docUpdate) {
        FieldUpdate fieldUpdate = docUpdate.getFieldUpdate(OUTLINKS_FIELD_NAME);
        if (fieldUpdate == null) { return Collections.emptySet(); }
        Set<String> links = new HashSet<>();
        ValueUpdate<Array<StringFieldValue>> valueUpdate = fieldUpdate.getValueUpdate(0);
        for (StringFieldValue value : valueUpdate.getValue()) {
            links.add(value.getString());
        }
        return links;
    }

    protected static Set<String> removeLinkFragment(Set<String> links) {
        // Remove fragment after # like vespa-cmdline-tools.html#vespa-model-inspect
        if (links == null) { return Collections.emptySet(); }
        return links.stream()
                .map(l -> l.replaceFirst("#.*$", ""))
                .filter(l -> !(l.isEmpty()))
                .collect(Collectors.toSet());
    }

    @Override
    public void deconstruct() {
        super.deconstruct();
        asyncSession.destroy();
    }
}
