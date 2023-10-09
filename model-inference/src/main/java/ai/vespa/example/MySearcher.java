// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example;

import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;


public class MySearcher extends Searcher {

    private final ModelsEvaluator modelsEvaluator;

    public MySearcher(ModelsEvaluator modelsEvaluator) {
        this.modelsEvaluator = modelsEvaluator;
    }

    @Override
    public Result search(Query query, Execution execution) {

        // Create evaluator
        FunctionEvaluator evaluator = modelsEvaluator.evaluatorOf("transformer");

        // Get input
        String inputString = query.properties().getString("input");

        // Convert to a Vespa tensor - and reshape to model expectations
        Tensor input = Tensor.from(TensorType.fromSpec("tensor<int8>(x[])"), inputString);
        input = Util.renameDimension(input, "x", "d1");
        input = Util.addDimension(input, "d0");

        // Evaluate model - output has 3 dimensions: d0,d1, and d2
        Tensor output = evaluator.bind("input", input).evaluate();

        // Retrieve embedding (values in the d2 dimension) of the first token
        Tensor embedding = Util.renameDimension(Util.slice(output, "d0:0,d1:0"), "d2", "x");

        // Add this tensor to query
        query.getRanking().getFeatures().put("query(embedding)", embedding);

        // Add a query to match all documents of type "mydoc"
        query.getModel().getQueryTree().setRoot(new WordItem("mydoc", "sddocname"));

        // Continue processing
        return execution.search(query);
    }

}
