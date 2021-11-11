// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.cloud.docsearch;

import com.yahoo.docproc.CallStack;
import com.yahoo.docproc.DocprocService;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.docproc.jdisc.metric.NullMetric;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentType;
import com.yahoo.document.DocumentUpdate;
import com.yahoo.document.config.DocumentmanagerConfig;
import com.yahoo.document.datatypes.Array;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.update.FieldUpdate;
import com.yahoo.documentapi.DocumentAccessParams;
import com.yahoo.documentapi.local.LocalDocumentAccess;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Collections;
import java.util.Set;

import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.CONTENT_FIELD_NAME;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.DOC_DOCUMENT_TYPE;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.INLINKS_FIELD_NAME;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.NAMESPACE_FIELD_NAME;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.OUTLINKS_FIELD_NAME;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.PATH_FIELD_NAME;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.TITLE_FIELD_NAME;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.canonicalizeLinks;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.getUniqueOutLinks;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.removeLinkFragment;


public class VespaDocLinksTest {

    private static DocprocService setupDocprocService(DocumentProcessor processor) {
        CallStack stack = new CallStack("default", new NullMetric());
        stack.addLast(processor);
        DocprocService service = new DocprocService("default");
        service.setCallStack(stack);
        service.setInService(true);
        return service;
    }

    private static Processing getProcessing(DocumentOperation... operations) {
        Processing processing = new Processing();
        for (DocumentOperation op : operations) {
            processing.addDocumentOperation(op);
        }
        return processing;
    }

    private static LocalDocumentAccess getDocumentAccess() {
        DocumentAccessParams dap = new DocumentAccessParams();
        dap.setDocumentmanagerConfig(new DocumentmanagerConfig.Builder().build());
        return new LocalDocumentAccess(dap);
    }

    private static DocumentType createDocType() {
        DocumentType type = new DocumentType(DOC_DOCUMENT_TYPE);
        type.addField(PATH_FIELD_NAME,      DataType.STRING);
        type.addField(NAMESPACE_FIELD_NAME, DataType.STRING);
        type.addField(TITLE_FIELD_NAME,     DataType.STRING);
        type.addField(CONTENT_FIELD_NAME,   DataType.STRING);
        type.addField(OUTLINKS_FIELD_NAME,  DataType.getArray(DataType.STRING));
        type.addField(INLINKS_FIELD_NAME,   DataType.getWeightedSet(DataType.STRING));
        return type;
    }

    @Test
    public void testNoOutLinksField() {
        Document doc = new Document(createDocType(), "id:open:doc::1");
        assert (getUniqueOutLinks(doc).equals(Collections.emptySet()));   // do no crash if field is missing in feed
    }

    @Test
    public void testFragments() {
        Set<String> links = Set.of(
                "#status-pages",
                "/documentation/reference/services-content.html#cluster-controller");
        assert (removeLinkFragment(links).equals(Set.of("/documentation/reference/services-content.html")));
    }

    @Test
    public void testPath() {
        assert(canonicalizeLinks(Path.of("/documentation/operations/admin-procedures.html"),
                Set.of("../reference/services.html",
                        "../config-sentinel.html",
                        "/documentation/reference/services-content.html"))
                .equals(Set.of("/documentation/reference/services.html",
                        "/documentation/config-sentinel.html",
                        "/documentation/reference/services-content.html")));
    }

    @Test
    public void testUpdate() {
        DocumentType docType = createDocType();

        DocumentUpdate update = new DocumentUpdate(docType, "id:open:doc::1");
        update.addFieldUpdate(FieldUpdate.createAssign(docType.getField(PATH_FIELD_NAME),      new StringFieldValue("/documentation/access-logging.html")));
        update.addFieldUpdate(FieldUpdate.createAssign(docType.getField(NAMESPACE_FIELD_NAME), new StringFieldValue("open")));
        update.addFieldUpdate(FieldUpdate.createAssign(docType.getField(TITLE_FIELD_NAME),     new StringFieldValue("Access Logging")));
        update.addFieldUpdate(FieldUpdate.createAssign(docType.getField(CONTENT_FIELD_NAME),   new StringFieldValue("Text here ...")));

        Array<StringFieldValue> inLinks = new Array<>(DataType.getArray(DataType.STRING));
        inLinks.add(new StringFieldValue("/documentation/operations/admin-procedures.html#fragment-to-be-removed"));
        inLinks.add(new StringFieldValue("/documentation/reference/advanced-indexing-language.html"));
        update.addFieldUpdate(FieldUpdate.createAssign(docType.getField(OUTLINKS_FIELD_NAME), inLinks));

        Processing processing = getProcessing(update);
        LocalDocumentAccess documentAccess = getDocumentAccess();
        DocprocService service = setupDocprocService(new OutLinksDocumentProcessor(documentAccess));

        service.getExecutor().process(processing);
    }
}
