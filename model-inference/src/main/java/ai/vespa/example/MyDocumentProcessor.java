// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example;

import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.tensor.Tensor;


public class MyDocumentProcessor extends DocumentProcessor {

    private final ModelsEvaluator modelsEvaluator;

    public MyDocumentProcessor(ModelsEvaluator modelsEvaluator) {
        this.modelsEvaluator = modelsEvaluator;
    }

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document document = put.getDocument();

                // Get tokens
                Tensor tokens = (Tensor) document.getFieldValue("tokens").getWrappedValue();

                // Reshape to expected input for model (d0[],d1[])
                Tensor input = Util.addDimension(Util.renameDimension(tokens, "tokens", "d1"), "d0");

                // Calculate embedding
                FunctionEvaluator evaluator = modelsEvaluator.evaluatorOf("transformer");
                Tensor output = evaluator.bind("input", input).evaluate();
                Tensor embedding = Util.renameDimension(Util.slice(output, "d0:0,d1:0"), "d2", "x");

                // Set embedding
                document.setFieldValue("embedding", new TensorFieldValue(embedding));

            }
        }
        return Progress.DONE;
    }
}
