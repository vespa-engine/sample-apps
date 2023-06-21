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
import java.util.Map;
import java.util.concurrent.Executor;

public class EmbeddingHandler extends ThreadedHttpRequestHandler {
    private ComponentRegistry<Embedder> availableEmbedders;
    private ObjectMapper jsonMapper;

    // Mappings fetched from https://cloud.vespa.ai/en/model-hub#using-e5-models
    private final Map<String, TensorType> modelTensorTypeMap = Map.of(
            "e5-small-v2", TensorType.fromSpec("tensor<float>(x[384])"),
            "e5-base-v2", TensorType.fromSpec("tensor<float>(x[768])"),
            "e5-large-v2", TensorType.fromSpec("tensor<float>(x[1024])"),
            "multilingual-e5-base", TensorType.fromSpec("tensor<float>(x[768])"),
            "minilm-l6-v2", TensorType.fromSpec("tensor<float>(x[384])"),
            "mpnet-base-v2", TensorType.fromSpec("tensor<float>(x[768])")
    );

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
        if (embedder == null) {
            return new HttpResponse(400) {
                @Override
                public void render(OutputStream outputStream) throws IOException {
                    outputStream.write(jsonMapper.writeValueAsBytes(Map.of("error", "Embedder '" + requestData.embedder()  + "' not found")));
                }
            };
        }

        TensorType type = modelTensorTypeMap.get(requestData.embedder());
        if (type == null) {
            return new HttpResponse(400) {
                @Override
                public void render(OutputStream outputStream) throws IOException {
                    outputStream.write(jsonMapper.writeValueAsBytes(Map.of("error", "TensorType for embedder '" + requestData.embedder()  + "' not found")));
                }
            };
        }

        Embedder.Context context = new Embedder.Context("");
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
