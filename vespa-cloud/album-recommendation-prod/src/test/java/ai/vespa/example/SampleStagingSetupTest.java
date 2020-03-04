package ai.vespa.example;

import ai.vespa.hosted.cd.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Before Vespa Cloud deploys an application to a production environment,
 * the new application will be tested in the upgrade phase (Staging Tests).
 * 
 * - First the previous application is deployed in an enviornment.
 * - Second the StagingSetup tests run.
 * - Third the running application in step one is upgraded.
 * - Fourth the StagingTest tests run.
 * 
 * By doing this we make sure the application works in an upgrade scenario,
 * and that e.g. changes to your search definitions will not cause any issues.
 * 
 * The test here is minimal, and only checks that the system is up and running.
 * See 
 */
@StagingSetup
class SampleStagingSetupTest {

    private final Endpoint endpoint = TestRuntime.get().deploymentToTest().endpoint("container");

    @Test
    void test_instance_running() {
        var result = endpoint.send(endpoint.request("/status.html"));
        assertEquals(200, result.statusCode());
    }

}