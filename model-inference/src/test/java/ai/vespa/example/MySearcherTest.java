// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

package ai.vespa.example;

import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.component.chain.Chain;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MySearcherTest {

    @Test
    public void testMySearcher() {
        ModelsEvaluator modelsEvaluator = ModelsEvaluatorTester.create("src/main/application/models");
        Query query = new Query("/search/?input=%7B%7Bx%3A0%7D%3A1%2C%7Bx%3A1%7D%3A2%2C%7Bx%3A2%7D%3A3%7D");  // {{x:0}:1,{x:1}:2,{x:2}:3}
        MySearcher mySearcher = new MySearcher(modelsEvaluator);

        Result r = execute(query, mySearcher);
        Optional<Tensor> t = query.getRanking().getFeatures().getTensor("query(embedding)");
        assertTrue(t.isPresent());

        Tensor expected = Tensor.from("tensor<float>(x[16]):[1.6495612, -1.0305781, -0.86149156, 1.180618, -1.2259362, 0.7068715, 0.057115182, 0.7203075, 0.95562154, 0.03489011, -1.3914255, -0.29500565, -0.33525327, -0.17927599, -1.4886415, 1.5026228]]");
        assertEquals(expected, t.get());
    }

    private static Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }

}
