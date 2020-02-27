package ai.vespa.example;

import ai.vespa.hosted.cd.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * This test implements a very minimal system test that is required for deploying
 * applications to production in Vespa Cloud.  System tests run in a separate test
 * environment before your application is eventually deployed to production.
 * 
 * This test is very minimal and does not really test anything.  See the testing
 * guide for writing more comprehensive application tests.
 */
@SystemTest
class SampleSystemTest {

    private final Endpoint endpoint = TestRuntime.get().deploymentToTest().endpoint("default");

    @Test
    void test_instance_running() {
        var result = endpoint.send(endpoint.request("/status.html"));
        assertEquals(200, result.statusCode());
    }
    
}