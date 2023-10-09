// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

package ai.vespa.example;

import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.application.Application;
import com.yahoo.application.Networking;
import com.yahoo.application.container.JDisc;
import com.yahoo.component.ComponentSpecification;
import com.yahoo.component.chain.Chain;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentType;
import com.yahoo.document.Field;
import com.yahoo.document.TensorDataType;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MyDocumentProcessorTest {

    @Test
    public void testMyDocumentProcessor() throws Exception {
        ModelsEvaluator modelsEvaluator = ModelsEvaluatorTester.create("src/main/application/models");
        MyDocumentProcessor docproc = new MyDocumentProcessor(modelsEvaluator);

        DocumentType myDocType = new DocumentType("mydoc");
        myDocType.addField(new Field("tokens", 1, new TensorDataType(TensorType.fromSpec("tensor<int8>(tokens[10])"))));
        myDocType.addField(new Field("embedding", 2, new TensorDataType(TensorType.fromSpec("tensor<float>(x[16])"))));

        Document doc = new Document(myDocType, "id:mydoc:mydoc::1");
        doc.setFieldValue("tokens", new TensorFieldValue(Tensor.from("tensor<int8>(tokens[10]):[1,2,3,0,0,0,0,0,0,0]")));
        Processing processing = new Processing();
        processing.addDocumentOperation(new DocumentPut(doc));
        docproc.process(processing);

        Tensor expected = Tensor.from("tensor<float>(x[16]):[1.6495612, -1.0305781, -0.86149156, 1.180618, -1.2259362, 0.7068715, 0.057115182, 0.7203075, 0.95562154, 0.03489011, -1.3914255, -0.29500565, -0.33525327, -0.17927599, -1.4886415, 1.5026228]]");
        assertEquals(expected, doc.getFieldValue("embedding").getWrappedValue());
    }

}
