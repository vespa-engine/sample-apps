// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping;

import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentType;
import com.yahoo.document.config.DocumentmanagerConfig;
import com.yahoo.document.datatypes.IntegerFieldValue;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.documentapi.DocumentAccessParams;
import com.yahoo.documentapi.local.LocalDocumentAccess;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class ReviewProcessorTest {

    private LocalDocumentAccess access;
    private ReviewProcessor processor;

    @BeforeEach
    public void setUp() {
        access = new LocalDocumentAccess(new DocumentAccessParams()
                .setDocumentmanagerConfig(new DocumentmanagerConfig.Builder().build()));
        processor = new ReviewProcessor(access);
    }

    @AfterEach
    public void tearDown() {
        processor.deconstruct();
        access.shutdown();
    }

    private static DocumentType createReviewType() {
        DocumentType type = new DocumentType("review");
        type.addField("asin", DataType.STRING);
        type.addField("stars", DataType.INT);
        return type;
    }

    private static Processing getProcessing(Document document) {
        Processing processing = new Processing();
        processing.addDocumentOperation(new DocumentPut(document));
        return processing;
    }

    @Test
    public void testReviewWithoutStarsIsDone() {
        Document review = new Document(createReviewType(), "id:review:review::B000-reviewer");
        review.setFieldValue("asin", new StringFieldValue("B000"));
        assertEquals(DocumentProcessor.Progress.DONE, processor.process(getProcessing(review)));
    }

    @Test
    public void testReviewWithoutAsinIsDone() {
        Document review = new Document(createReviewType(), "id:review:review::B000-reviewer");
        review.setFieldValue("stars", new IntegerFieldValue(5));
        assertEquals(DocumentProcessor.Progress.DONE, processor.process(getProcessing(review)));
    }

}
