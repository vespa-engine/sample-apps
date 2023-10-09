// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples;

import com.yahoo.component.chain.Chain;
import com.yahoo.processing.Processor;
import com.yahoo.processing.Request;
import com.yahoo.processing.Response;
import com.yahoo.processing.execution.Execution;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;


/**
 * Unit test of the example processor.
 * If ExampleProcessorConfig is not found you need to run mvn install on this project.
 */
public class ExampleProcessorTest {

    @Test
    public void testThatResultContainsHelloWorld() {
        ExampleProcessorConfig.Builder config = new ExampleProcessorConfig.Builder().message("Hello, processor!");
        Processor processor = new ExampleProcessor(new ExampleProcessorConfig(config));

        Response response = newExecution(processor).process(new Request());
        assertEquals("Hello, processor!", response.data().get(0).toString());
    }

    private static Execution newExecution(Processor... processors) {
        return Execution.createRoot(new Chain<>(processors), 0, Execution.Environment.createEmpty());
    }

}
