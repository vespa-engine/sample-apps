// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

package ai.vespa.example;

import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.searchlib.rankingexpression.parser.ParseException;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ModelsTest {

    @Test
    public void testModels() throws ParseException {
        ModelsEvaluator modelsEvaluator = ModelsEvaluatorTester.create("src/main/application/models");

        // The "models" directory contains 2 ONNX models: "transformer" and "pairwise_ranker"
        assertEquals(2, modelsEvaluator.models().size());
        assertTrue(modelsEvaluator.models().containsKey("transformer"));
        assertTrue(modelsEvaluator.models().containsKey("pairwise_ranker"));

        testTransformerModel(modelsEvaluator);
        testPairwiseRanker(modelsEvaluator);
    }

    private void testTransformerModel(ModelsEvaluator modelsEvaluator) throws ParseException {
        FunctionEvaluator transformer = modelsEvaluator.evaluatorOf("transformer");

        // The transformer model has one input, named "input" and has type "tensor(d0[],d1[])"
        assertEquals(1, transformer.function().arguments().size());
        assertEquals("input", transformer.function().arguments().get(0));
        assertEquals("tensor(d0[],d1[])", transformer.function().argumentTypes().get("input").toString());

        // Test the model
        Tensor input = Tensor.from("tensor<float>(d0[1],d1[3]):[1,2,3]");
        Tensor output = transformer.bind("input", input).evaluate();

        // Check the resulting type type. We send in 1 batch dimension (d0),
        // 3 "tokens" (d1) and expect a result where each token has embedding
        // size of 16 (d3):
        assertEquals("tensor<float>(d0[1],d1[3],d2[16])", output.type().toString());

        // Check the value of the first cell
        assertEquals(1.64956, output.get(TensorAddress.of(0,0,0)), 1e-5);
    }

    private void testPairwiseRanker(ModelsEvaluator modelsEvaluator) {
        FunctionEvaluator ranker = modelsEvaluator.evaluatorOf("pairwise_ranker");

        // The ranker has 3 inputs:
        assertEquals(3, ranker.function().arguments().size());
        assertTrue(ranker.function().argumentTypes().containsKey("query"));
        assertTrue(ranker.function().argumentTypes().containsKey("doc1"));
        assertTrue(ranker.function().argumentTypes().containsKey("doc2"));
        assertEquals("tensor<float>(d0[],d1[16])", ranker.function().argumentTypes().get("query").toString());
        assertEquals("tensor<float>(d0[],d1[16])", ranker.function().argumentTypes().get("doc1").toString());
        assertEquals("tensor<float>(d0[],d1[16])", ranker.function().argumentTypes().get("doc2").toString());

        // Test the model
        Tensor query = Tensor.from("tensor<float>(d0[1],d1[16]):[0.11,0.94,0.04,0.86,0.75,0.88,0.73,0.21,0.67,0.68,0.82,0.93,0.88,0.66,0.66,0.91]");
        Tensor doc1 =  Tensor.from("tensor<float>(d0[1],d1[16]):[0.02,0.98,0.31,0.28,0.44,0.17,0.18,0.62,0.78,0.57,0.89,0.39,0.69,0.16,0.70,0.79]");
        Tensor doc2 =  Tensor.from("tensor<float>(d0[1],d1[16]):[0.32,0.33,0.02,0.90,0.85,0.93,0.89,0.12,0.06,0.08,0.26,0.62,0.40,0.94,0.64,0.23]");
        Tensor output = ranker.bind("query", query).bind("doc1", doc1).bind("doc2", doc2).evaluate();
        assertEquals("tensor<float>(d0[1],d1[1])", output.type().toString());
        assertEquals(0.53560, output.get(TensorAddress.of(0,0)), 1e-5);
    }

}
