// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.container.StatisticsConfig;
import com.yahoo.docproc.CallStack;
import com.yahoo.docproc.DocprocService;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.docproc.jdisc.metric.NullMetric;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentType;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.statistics.StatisticsImpl;

import static ai.vespa.example.album.ProductTypeRefinerDocProc.MUSIC_DOCUMENT_TYPE;
import static ai.vespa.example.album.ProductTypeRefinerDocProc.PRODUCT_TYPE_FIELD_NAME;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ProductTypeRefinerDocProcTest {

    private static DocprocService setupDocprocService(DocumentProcessor processor) {
        CallStack stack = new CallStack("default", new StatisticsImpl(new StatisticsConfig(new StatisticsConfig.Builder())), new NullMetric());
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

    private static DocumentType createMusicType() {
        DocumentType type = new DocumentType(MUSIC_DOCUMENT_TYPE);
        type.addField(PRODUCT_TYPE_FIELD_NAME, DataType.STRING);
        return type;
    }

    @Test
    public void testRefinement() {
        Document doc = new Document(createMusicType(), "id:mynamespace:music::1");
        doc.setFieldValue(PRODUCT_TYPE_FIELD_NAME,
                new StringFieldValue("Media > Music & Sound Recordings > Music Cassette Tapes"));

        Processing p = getProcessing(new DocumentPut(doc));
        DocprocService service = setupDocprocService(new ProductTypeRefinerDocProc());
        service.getExecutor().process(p);

        assertEquals("Media | Music & Sound Recordings | Music Cassette Tapes",
                doc.getFieldValue(PRODUCT_TYPE_FIELD_NAME).toString());
    }
}
