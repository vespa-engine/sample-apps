// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.models.evaluation;

/*
 * Mock the FunctionEvaluator that has package private visibility of it's constructor
 */
public class MockFunctionEvaluator extends FunctionEvaluator {
    public MockFunctionEvaluator() {
        super(null, null);
    }
}
