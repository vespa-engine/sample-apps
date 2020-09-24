// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.cloud.docsearch;

import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentType;
import com.yahoo.document.datatypes.Array;
import com.yahoo.document.datatypes.StringFieldValue;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Collections;
import java.util.Set;

import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.DOC_DOCUMENT_TYPE;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.OUTLINKS_FIELD_NAME;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.PATH_FIELD_NAME;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.removeLinkFragment;
import static ai.vespa.cloud.docsearch.OutLinksDocumentProcessor.canonicalizeLinks;


public class VespaDocLinksTest {

    private static DocumentType createDocType() {
        DocumentType type = new DocumentType(DOC_DOCUMENT_TYPE);
        type.addField(PATH_FIELD_NAME, DataType.STRING);
        type.addField(OUTLINKS_FIELD_NAME, DataType.getArray(DataType.STRING));
        return type;
    }

    @Test
    public void testFragments() {
        Document doc = new Document(createDocType(), "id:open:doc::1");
        Array<StringFieldValue> strArr = new Array<>(doc.getField(OUTLINKS_FIELD_NAME).getDataType());
        strArr.add(new StringFieldValue("#status-pages"));
        strArr.add(new StringFieldValue("/documentation/reference/services-content.html#cluster-controller"));
        doc.setFieldValue(OUTLINKS_FIELD_NAME, strArr);
        assert (removeLinkFragment(doc).equals(Set.of("/documentation/reference/services-content.html")));
    }

    @Test
    public void testNoOutLinksField() {
        Document doc = new Document(createDocType(), "id:open:doc::1");
        assert (removeLinkFragment(doc).equals(Collections.emptySet()));   // do no crash if field is missing in feed
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
}
