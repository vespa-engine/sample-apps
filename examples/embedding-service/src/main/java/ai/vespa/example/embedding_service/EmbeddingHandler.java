package ai.vespa.example.embedding_service;

import com.fasterxml.jackson.databind.ObjectMapper;

import com.yahoo.component.provider.ComponentRegistry;
import com.yahoo.container.jdisc.HttpRequest;
import com.yahoo.container.jdisc.HttpResponse;
import com.yahoo.container.jdisc.ThreadedHttpRequestHandler;
import com.yahoo.component.annotation.Inject;
import com.yahoo.language.process.Embedder;
import com.yahoo.tensor.TensorType;

import java.io.IOException;
import java.io.OutputStream;
import java.util.concurrent.Executor;

public class EmbeddingHandler extends ThreadedHttpRequestHandler {
    private ComponentRegistry<Embedder> availableEmbedders;
    private ObjectMapper jsonMapper;

    @Inject
    public EmbeddingHandler(Executor executor, ComponentRegistry<Embedder> embedders) {
        super(executor);
        availableEmbedders = embedders;
        jsonMapper = new ObjectMapper();
    }

    @Override
    public HttpResponse handle(HttpRequest httpRequest) {
        Data requestData = parseRequestJson(httpRequest);

        Embedder embedder = availableEmbedders.getComponent(requestData.embedder());
        Embedder.Context context = new Embedder.Context("");
        TensorType type = TensorType.fromSpec("tensor<float>(x[384])");
        String embedding = embedder.embed(requestData.text(), context, type).toString();

        Data responseData = new Data(requestData.text(), requestData.embedder(), embedding);

        return new HttpResponse(200) {
            @Override
            public void render(OutputStream outputStream) throws IOException {
                outputStream.write(jsonMapper.writeValueAsBytes(responseData));
            }
        };
    }

    private Data parseRequestJson(HttpRequest httpRequest) {
        byte[] json = new byte[0];
        try {
            json = httpRequest.getData().readAllBytes();
            return jsonMapper.readValue(json, Data.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}

record Data(String embedder, String text, String embedding) {}
