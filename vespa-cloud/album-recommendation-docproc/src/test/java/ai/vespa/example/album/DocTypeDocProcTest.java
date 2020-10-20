// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.application.Application;
import com.yahoo.application.Networking;
import com.yahoo.application.container.JDisc;
import com.yahoo.component.ComponentSpecification;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentType;
import com.yahoo.document.DocumentTypeManager;
import com.yahoo.document.config.DocumentmanagerConfig;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static ai.vespa.example.album.DocTypeDocumentProcessor.ARTIST_FIELD_NAME;
import static ai.vespa.example.album.DocTypeDocumentProcessor.MUSIC_DOCUMENT_TYPE;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class DocTypeDocProcTest {

    private static DocumentType createMusicDocumentType() {
        DocumentType type = new DocumentType(MUSIC_DOCUMENT_TYPE);
        type.addField(ARTIST_FIELD_NAME, DataType.STRING);
        return type;
    }

    /**
     * Test the function to create a Document from JSON
     */
    @Test
    public void testCreateOperationFromJson() {
        DocTypeDocumentProcessor docProc = new DocTypeDocumentProcessor();
        DocumentType docType = createMusicDocumentType();
        DocumentTypeManager typeMgr = new DocumentTypeManager(new DocumentmanagerConfig.Builder().build());
        typeMgr.registerDocumentType(docType);
        DocumentOperation op = docProc.createDocFromJson(typeMgr,"id:mynamespace:" + MUSIC_DOCUMENT_TYPE + "::1");

        assertEquals("put of document id:mynamespace:" + MUSIC_DOCUMENT_TYPE + "::1", op.toString());
    }

    /**
     * Using Application, one can load the DocumentType, aka schema, directly,
     * and set up the processing chain programmatically
     */
    @Test
    public void testDocTypeUsingApplication () throws Exception {
        Application app = Application.fromBuilder(new Application.Builder()
                .documentType(MUSIC_DOCUMENT_TYPE, Files.readString(Path.of("src/main/application/schemas/" + MUSIC_DOCUMENT_TYPE + ".sd")))
                .container("myContainer", new Application.Builder.Container()
                        .documentProcessor("myChain", DocTypeDocumentProcessor.class))
                .networking(Networking.disable));
        JDisc container = app.getJDisc("myContainer");

        Processing processing = new Processing();
        Document doc = new Document(container.documentProcessing().getDocumentTypes().get(MUSIC_DOCUMENT_TYPE),
                "id:mynamespace:" + MUSIC_DOCUMENT_TYPE + "::1");
        processing.addDocumentOperation(new DocumentPut(doc));
        container.documentProcessing().processOnce(ComponentSpecification.fromString("myChain"), processing);

        assertEquals("Elvis", doc.getFieldValue(ARTIST_FIELD_NAME).toString());
    }
}
