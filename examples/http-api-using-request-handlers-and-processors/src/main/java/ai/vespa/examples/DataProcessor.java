// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples;

import com.google.inject.Inject;
import com.yahoo.component.provider.ListenableFreezableClass;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentType;
import com.yahoo.document.DocumentTypeManager;
import com.yahoo.processing.Processor;
import com.yahoo.processing.Request;
import com.yahoo.processing.Response;
import com.yahoo.processing.execution.Execution;
import com.yahoo.processing.response.ArrayDataList;
import com.yahoo.processing.response.Data;
import com.yahoo.processing.response.DataList;
import com.yahoo.yolean.chain.After;
import com.yahoo.yolean.chain.Provides;

import java.util.ArrayList;
import java.util.List;

/**
 * A processor making a nested result sets of "normalized" strings from the
 * request property {@link AnnotatingProcessor.DemoProperty#NAME}.
 */
@Provides(DataProcessor.DemoData.NAME)
@After(AnnotatingProcessor.DemoProperty.NAME)
public class DataProcessor extends Processor {

    public static class DemoData extends ListenableFreezableClass implements Data {
        public static final String NAME = "DemoData";

        private final Request request;
        private final String content;

        DemoData(Request request, String content) {
            this.request = request;
            this.content = content;
        }

        @Override
        public Request request() {
            return request;
        }

        public String content() {
            return content;
        }

        public String toString() {
            return NAME + "(\"" + content + "\")";
        }
    }

    private final DemoComponent termChecker;
    private final DemoFreezableComponent docApiClient;
    private final DocumentType musicType;

    @Inject
    public DataProcessor(DemoComponent termChecker,
                         DemoFreezableComponent docApiClient,
                         DocumentTypeManager documentTypeManager) {
        this.termChecker = termChecker;
        this.docApiClient = docApiClient;
        this.musicType = documentTypeManager.getDocumentType("music");
    }

    @Override
    public Response process(Request request, Execution execution) {
        Response r = new Response(request);
        @SuppressWarnings("unchecked")
        DataList<Data> current = r.data();
        DataList<Data> previous = null;
        String exampleProperty = request.properties().getString(DemoHandler.REQUEST_URI);
        Object o = request.properties().get(AnnotatingProcessor.DemoProperty.NAME_AS_COMPOUND);
        List<Document> newDocuments = new ArrayList<>();


        if (exampleProperty != null) {
            current.add(new DemoData(request, exampleProperty));
        }

        if (o instanceof AnnotatingProcessor.DemoProperty) {
            // create a nested result set with a level for each term
            for (String s : ((AnnotatingProcessor.DemoProperty) o).terms()) {
                String normalized = termChecker.normalize(s);
                DemoData data = new DemoData(request, normalized);

                if (current == null) {
                    current = ArrayDataList.create(request);
                }
                current.add(data);
                if (previous != null) {
                    previous.add(current);
                }
                previous = current;
                current = null;

                // For Demo purposes lets create a new Document named after each given term
                newDocuments.add(new Document(this.musicType, "id:default:music::" + normalized));
            }
        }
        Processing processing = new Processing();
        for (Document document : newDocuments) {
            DocumentPut docPut = new DocumentPut(document);
            processing.addDocumentOperation(docPut);
        }
        docApiClient.syncProcess(processing);
        return r;
    }

}
