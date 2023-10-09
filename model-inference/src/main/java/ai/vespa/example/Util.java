// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example;

import com.yahoo.searchlib.rankingexpression.RankingExpression;
import com.yahoo.searchlib.rankingexpression.evaluation.MapContext;
import com.yahoo.searchlib.rankingexpression.evaluation.TensorValue;
import com.yahoo.searchlib.rankingexpression.parser.ParseException;
import com.yahoo.tensor.Tensor;

import java.util.Map;

public class Util {

    public static Tensor renameDimension(Tensor tensor, String from, String to) {
        return tensor.rename(from, to);
    }

    public static Tensor addDimension(Tensor tensor, String name) {
        return tensor.multiply(Tensor.from("tensor(" + name + "[1]):[1]"));
    }

    public static Tensor slice(Tensor tensor, String sliceExpr) {
        return evaluate("t{" + sliceExpr + "}", "t", tensor);
    }

    public static Tensor evaluate(String expression) {
        return evaluate(expression, new MapContext());
    }

    public static Tensor evaluate(String expression, String name, Tensor value) {
        return evaluate(expression, new MapContext(Map.of(name, new TensorValue(value))));
    }

    public static Tensor evaluate(String expression, MapContext context) {
        try {
            return new RankingExpression(expression).evaluate(context).asTensor();
        } catch (ParseException e) {
            throw new RuntimeException("Unable to parse ranking expression: " + expression, e);
        }
    }

}
