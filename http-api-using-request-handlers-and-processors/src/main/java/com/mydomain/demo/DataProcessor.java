// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.mydomain.demo;

import com.google.inject.Inject;
import com.yahoo.component.provider.ListenableFreezableClass;
import com.yahoo.docproc.DocprocService;
import com.yahoo.docproc.DocumentProcessor.Progress;
import com.yahoo.docproc.Processing;
import com.yahoo.docproc.jdisc.DocumentProcessingHandler;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentType;
import com.yahoo.processing.Processor;
import com.yahoo.processing.Request;
import com.yahoo.processing.Response;
import com.yahoo.processing.execution.Execution;
import com.yahoo.processing.response.ArrayDataList;
import com.yahoo.processing.response.Data;
import com.yahoo.processing.response.DataList;
import com.yahoo.yolean.chain.After;
import com.yahoo.yolean.chain.Provides;

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
    private final DocumentProcessingHandler docHandler;

    @Inject
    public DataProcessor(DemoComponent termChecker, DocumentProcessingHandler docHandler) {
        this.termChecker = termChecker;
        this.docHandler = docHandler;
    }

    @Override
    public Response process(Request request, Execution execution) {
        Response r = new Response(request);
        @SuppressWarnings("unchecked")
        DataList<Data> current = r.data();
        DataList<Data> previous = null;
        String exampleProperty = request.properties().getString(DemoHandler.REQUEST_URI);
        Object o = request.properties().get(AnnotatingProcessor.DemoProperty.NAME_AS_COMPOUND);


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
            }
        }

        /*
        NOTE: As example, we are going to attempt storing a new Document (hardcoded data)
        On a Real application I assume one should be able of creating/updating any number
        of documents from here by creating corresponding DocumentOperation(s)
        */

        DocumentType type = this.docHandler.getDocumentTypeManager().getDocumentType("music");
        Document document = new Document(type, "id:default:music::10");
        document.setFieldValue("title", "My Test Title");
        DocumentPut docPut = new DocumentPut(document);

        Processing proc = com.yahoo.docproc.Processing.of(docPut);

        DocprocService docProcService = this.docHandler.getDocprocServiceRegistry()
            .getComponent("default");
        proc.setDocprocServiceRegistry(this.docHandler.getDocprocServiceRegistry());

        Progress progress = docProcService.getExecutor().processUntilDone(proc);

        //
        // At this point, I'll expect that document is created, but it not, this returns 404:
        //    curl -v "http://localhost:8080/document/v1/default/music/docid/10"
        //

        return r;
    }

}
