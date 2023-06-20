package ai.vespa.example.embedding_service;

import com.fasterxml.jackson.core.JsonProcessingException;
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
        byte[] json = new byte[0];
        try {
            json = httpRequest.getData().readAllBytes();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        Data data;
        try {
            data = jsonMapper.readValue(json, Data.class);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        Embedder embedder = availableEmbedders.getComponent(data.embedder);
        Embedder.Context context = new Embedder.Context("");
        TensorType type = TensorType.fromSpec("tensor<float>(x[384])");
        data.embedding = embedder.embed(data.text, context, type).toString();

        return new HttpResponse(200) {
            @Override
            public void render(OutputStream outputStream) throws IOException {
                outputStream.write(jsonMapper.writeValueAsBytes(data));
            }
        };
    }
}

class Data {
    public String embedder;
    public String text; // Before embedding
    public String embedding; // After embedding
}
