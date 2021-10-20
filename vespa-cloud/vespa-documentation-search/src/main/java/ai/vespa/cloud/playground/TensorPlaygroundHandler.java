// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.cloud.playground;

import com.yahoo.container.jdisc.HttpRequest;
import com.yahoo.container.jdisc.HttpResponse;
import com.yahoo.container.jdisc.ThreadedHttpRequestHandler;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.URI;
import java.util.concurrent.Executor;


public class TensorPlaygroundHandler extends ThreadedHttpRequestHandler {

    public TensorPlaygroundHandler(Executor executor) {
        super(executor);
    }

    @Override
    public HttpResponse handle(HttpRequest httpRequest) {
        if (!"playground".equals(getPath(httpRequest, 0))) {
            return error(httpRequest, "Invalid path");
        }
        if ("eval".equals(getPath(httpRequest, 1))) {
            return eval(httpRequest);
        }
        return error(httpRequest, "Invalid path");
    }


    private HttpResponse eval(HttpRequest httpRequest) {
        String json = getParam(httpRequest, "json");
        if (json == null) {
            return error(httpRequest, "Eval operation requires an expression to evaluate");
        }
        return new HttpResponse(200) {
            @Override
            public void render(OutputStream stream) throws IOException {
                String result = ProtonExpressionEvaluator.evaluate(json);
                try (PrintWriter writer = new PrintWriter(stream)) {
                    writer.print(result);
                }
            }
            @Override
            public String getContentType() {
                return "application/json";
            }
        };
    }

    private HttpResponse error(HttpRequest httpRequest, String msg) {
        return error(httpRequest, msg, 400);
    }

    private HttpResponse error(HttpRequest httpRequest, String msg, int status) {
        return new HttpResponse(status) {
            @Override
            public void render(OutputStream stream) throws IOException {
                try (PrintWriter writer = new PrintWriter(stream)) {
                    writer.print(msg);
                }
            }
        };
    }

    private String getPath(HttpRequest httpRequest) {
        return getPath(httpRequest, null);
    }

    private String getPath(HttpRequest httpRequest, String defaultPath) {
        URI uri = httpRequest.getUri();
        String path = uri.getPath();
        if (path == null || path.length() == 0) {
            return defaultPath;
        }
        return path.startsWith("/") ? path.substring(1) : path;
    }

    private String getPath(HttpRequest httpRequest, int level) {
        String[] path = getPath(httpRequest).split("/");
        return (level >= 0 && level < path.length) ? path[level].toLowerCase() : null;
    }

    private String getParam(HttpRequest httpRequest, String param) {
        return httpRequest.getProperty(param);
    }

}
