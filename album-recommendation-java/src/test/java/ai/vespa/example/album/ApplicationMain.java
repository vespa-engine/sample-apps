// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.application.Application;
import com.yahoo.application.Networking;

import java.nio.file.FileSystems;
import java.util.Objects;


/**
 * This uses the Application class to set up a container instance of this application
 * in this JVM. All other aspects of the application package than a single container
 * cluster is ignored. This is useful for e.g starting a container instance in your IDE
 * and serving real HTTP requests for interactive debugging.
 * <p>
 * After running main you can e.g open
 * http://localhost:8080/search/?query=title:foo&tracelevel=2
 * in a browser.
 */
public class ApplicationMain {

    public static void main(String[] args) throws Exception {
        try (Application app = Application.fromApplicationPackage(FileSystems.getDefault().getPath("src/main/application"),
                                                                  Networking.enable)) {
            Objects.requireNonNull(app);
            Thread.sleep(Long.MAX_VALUE);
        }
    }

}
