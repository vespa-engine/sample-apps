// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples;

import com.yahoo.application.Networking;
import org.junit.jupiter.api.Test;

import java.nio.file.FileSystems;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

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
            app.getClass(); // throws NullPointerException
            Thread.sleep(Long.MAX_VALUE);
        }
    }
}
