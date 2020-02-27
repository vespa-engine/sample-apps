package ai.vespa.example;

import ai.vespa.hosted.cd.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

@StagingTest
class SampleStagingTeset {

    private final Endpoint endpoint = TestRuntime.get().deploymentToTest().endpoint("default");

    @Test
    void test_instance_running() {
        var result = endpoint.send(endpoint.request("/status.html"));
        assertEquals(200, result.statusCode());
    }

}