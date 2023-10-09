// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespasamples;

import com.yahoo.application.Networking;
import org.junit.jupiter.api.Test;

import java.nio.file.FileSystems;
import java.util.Objects;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * After running main you can e.g. open
 * http://localhost:8080/search/?query=foo&tracelevel=2
 * in a browser.
 */
public class ApplicationMain {

    @Test
    public void runFromMaven() throws Exception {
        assumeTrue(Boolean.valueOf(System.getProperty("isMavenSurefirePlugin")));
        main(null);
    }

    public static void main(String[] args) throws Exception {
        try (com.yahoo.application.Application app = com.yahoo.application.Application.fromApplicationPackage(
                FileSystems.getDefault().getPath("src/main/application"),
                Networking.enable)) {
            Objects.requireNonNull(app);
            Thread.sleep(Long.MAX_VALUE);
        }
    }
}
