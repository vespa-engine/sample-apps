// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.application.Application;
import com.yahoo.application.container.JDisc;
import com.yahoo.component.ComponentSpecification;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.StringFieldValue;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static ai.vespa.example.album.ProductTypeRefinerDocProc.MUSIC_DOCUMENT_TYPE;
import static ai.vespa.example.album.ProductTypeRefinerDocProc.PRODUCT_TYPE_FIELD_NAME;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ProductTypeRefinerDocProcTest {

    @Test
    public void testRefinement () throws Exception {
        Application app = Application.fromBuilder(
                new Application.Builder()
                        .documentType(MUSIC_DOCUMENT_TYPE, Files.readString(Path.of("src/main/application/schemas/" + MUSIC_DOCUMENT_TYPE + ".sd")))
                        .container("myContainer", new Application.Builder.Container()
                                .documentProcessor("myChain", ProductTypeRefinerDocProc.class)
                        ));
        JDisc container = app.getJDisc("myContainer");

        Processing processing = new Processing();
        Document doc = new Document(container.documentProcessing().getDocumentTypes().get(MUSIC_DOCUMENT_TYPE),
                                    "id:mynamespace:" + MUSIC_DOCUMENT_TYPE + "::a-head-full-of-dreams");
        doc.setFieldValue(PRODUCT_TYPE_FIELD_NAME,
                          new StringFieldValue("Media > Music & Sound Recordings > Music Cassette Tapes"));

        processing.addDocumentOperation(new DocumentPut(doc));
        container.documentProcessing().processOnce(ComponentSpecification.fromString("myChain"), processing);

        assertEquals("Media | Music & Sound Recordings | Music Cassette Tapes",
                     doc.getFieldValue(PRODUCT_TYPE_FIELD_NAME).toString());
    }
}
