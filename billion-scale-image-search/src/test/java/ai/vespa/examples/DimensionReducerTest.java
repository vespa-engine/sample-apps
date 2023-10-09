// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples;

import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
public class DimensionReducerTest {

    @Test
    public void test_dimension_reduction() {
        ModelsEvaluator eval = ModelsEvaluatorTester.create("src/main/application/models/");
        DimensionReducer reducer = new DimensionReducer(eval);
        TensorType type = new TensorType.Builder(TensorType.Value.FLOAT).indexed("x", 768).build();
        IndexedTensor.Builder builder = IndexedTensor.Builder.of(type);
        for (int j = 0; j < 768; j++)
            builder.cell(1.0,j);
        Tensor reduced = reducer.reduce(builder.build());
        assertEquals(128, reduced.size());
        String stringType = reduced.toAbbreviatedString().split(":")[0];
        assertEquals("tensor<float>(x[128])", stringType);
    }
}
