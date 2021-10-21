// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.frontend;

import com.yahoo.container.jdisc.HttpRequest;
import com.yahoo.container.jdisc.HttpResponse;
import com.yahoo.container.jdisc.ThreadedHttpRequestHandler;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.URI;
import java.util.concurrent.Executor;

public class FrontendHandler extends ThreadedHttpRequestHandler {

    public FrontendHandler(Executor executor) {
        super(executor);
    }

    @Override
    public HttpResponse handle(HttpRequest httpRequest) {
        return file(httpRequest);
    }

    private HttpResponse file(HttpRequest httpRequest) {
        final String filename = getPath(httpRequest, "frontend/index.html");
        if (!resourceExists(filename)) {
            return error("Error retrieving " + filename, 404);
        }
        return new HttpResponse(200) {
            @Override
            public void render(OutputStream outputStream) throws IOException {
                try(InputStream inputStream = resourceAsStream(filename)) {
                    copy(inputStream, outputStream);
                }
            }
            @Override
            public String getContentType() {
                return contentType(filename);
            }
        };
    }

    private static void copy(InputStream source, OutputStream sink) throws IOException {
        byte[] buf = new byte[4096];
        int n;
        while ((n = source.read(buf)) > 0) {
            sink.write(buf, 0, n);
        }
    }

    private static String contentType(String fname) {
        switch (fname.substring(fname.lastIndexOf(".") + 1)) {
            case "html": return "text/html";
            case "css":  return "text/css";
            case "js":   return "application/javascript";
            case "json": return "application/json";
        }
        return "text/plain";
    }

    private boolean resourceExists(String filename) {
        return getClass().getClassLoader().getResource(filename) != null;
    }

    private InputStream resourceAsStream(String filename) {
        return getClass().getClassLoader().getResourceAsStream(filename);
    }

    private HttpResponse error(String msg, int status) {
        return new HttpResponse(status) {
            @Override
            public void render(OutputStream stream) {
                try (PrintWriter writer = new PrintWriter(stream)) {
                    writer.print(msg);
                }
            }
        };
    }

    private String getPath(HttpRequest httpRequest, String defaultPath) {
        URI uri = httpRequest.getUri();
        String path = uri.getPath();
        if (path == null || path.length() == 0) {
            return defaultPath;
        }
        return path.startsWith("/") ? path.substring(1) : path;
    }

}
