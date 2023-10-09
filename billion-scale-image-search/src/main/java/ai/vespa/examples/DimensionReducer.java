// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples;

import ai.vespa.models.evaluation.ModelsEvaluator;
import com.google.inject.Inject;
import com.yahoo.component.AbstractComponent;
import com.yahoo.tensor.Tensor;

public class DimensionReducer extends AbstractComponent {

    private final ModelsEvaluator evaluator;

    @Inject
    public DimensionReducer(ModelsEvaluator evaluator) {
        this.evaluator = evaluator;
    }

    public Tensor reduce(Tensor tensor) {
        Tensor input = tensor.rename("x","d0");
        Tensor reduced = evaluator.evaluatorOf("pca_transformer").bind("vector", input).evaluate();
        return reduced.rename("d0", "x").l2Normalize("x");
    }

}
