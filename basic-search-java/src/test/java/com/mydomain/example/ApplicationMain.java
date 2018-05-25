// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.mydomain.example;

import com.yahoo.application.Networking;
import org.junit.Test;

import java.nio.file.FileSystems;

import static org.junit.Assume.assumeTrue;

/**
 * This uses the Application class to set up a container instance of this application
 * in this JVM. All other aspects of the application package than a single container
 * cluster is ignored. This is useful for e.g starting a container instance in your IDE
 * and serving real HTTP requests for interactive debugging.
 */
public class ApplicationMain {

    public static void main(String[] args) throws Exception {
        try (com.yahoo.application.Application app = com.yahoo.application.Application.fromApplicationPackage(
                FileSystems.getDefault().getPath("src/main/application"),
                Networking.enable)) {
            app.getClass(); // throws NullPointerException
            Thread.sleep(Long.MAX_VALUE);
        }
    }

}
