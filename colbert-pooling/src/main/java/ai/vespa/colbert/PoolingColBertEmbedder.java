// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.colbert;

import ai.vespa.modelintegration.evaluator.OnnxEvaluator;
import ai.vespa.modelintegration.evaluator.OnnxEvaluatorOptions;
import ai.vespa.modelintegration.evaluator.OnnxRuntime;
import com.yahoo.ai.vespa.colbert.PoolingColbertEmbedderConfig;
import com.yahoo.component.AbstractComponent;
import com.yahoo.component.annotation.Inject;
import com.yahoo.language.huggingface.HuggingFaceTokenizer;
import com.yahoo.language.process.Embedder;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.yahoo.language.huggingface.ModelInfo.TruncationStrategy.LONGEST_FIRST;

/**
 * A ColBERT embedder with optional hierarchical token pooling.
 * <p>
 * Constructs input sequences matching <a href="https://github.com/lightonai/pylate">pylate</a>'s
 * ColBERT encoding: {@code [CLS] [D] search_document: <text> [SEP]} for documents and
 * {@code [CLS] [Q] search_query: <text> [MASK]...} for queries.  The {@code search_document:} /
 * {@code search_query:} prefixes are configurable via {@code prependDocument} / {@code prependQuery}.
 * <p>
 * When {@code poolFactor > 1}, applies Ward's agglomerative clustering
 * (see {@link HierarchicalTokenPooling}) to merge semantically similar tokens,
 * reducing the multi-vector representation size while preserving retrieval quality.
 * <p>
 * Configured via {@code pooling-colbert-embedder.def}:
 * <ul>
 *   <li>{@code poolFactor=0} — standard ColBERT (no pooling)</li>
 *   <li>{@code poolFactor=2} — merge semantically similar tokens, keep ~half the vectors</li>
 *   <li>{@code poolFactor=3} — keep roughly a third, etc.</li>
 * </ul>
 */
public class PoolingColBertEmbedder extends AbstractComponent implements Embedder {

    private static final String PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

    private final Embedder.Runtime runtime;
    private final String inputIdsName;
    private final String attentionMaskName;
    private final String outputName;
    private final HuggingFaceTokenizer tokenizer;
    private final OnnxEvaluator evaluator;
    private final int maxTransformerTokens;
    private final int maxQueryTokens;
    private final int maxDocumentTokens;
    private final long startSequenceToken;
    private final long endSequenceToken;
    private final long maskSequenceToken;
    private final long padSequenceToken;
    private final long querySequenceToken;
    private final long documentSequenceToken;
    private final Set<Long> skipTokens;
    private final int poolFactor;
    private final String prependQuery;
    private final String prependDocument;

    record TransformerInput(List<Long> inputIds, List<Long> attentionMask) {}
    record EmbedderCacheKey(String embedderId, Object embeddedValue) {}
    record EmbeddingResult(int inputIdSize, Map<String, Tensor> outputs) {}

    @Inject
    public PoolingColBertEmbedder(OnnxRuntime onnx, Embedder.Runtime runtime, PoolingColbertEmbedderConfig config) {
        this.runtime = runtime;
        this.inputIdsName = config.transformerInputIds();
        this.attentionMaskName = config.transformerAttentionMask();
        this.outputName = config.transformerOutput();
        this.maxTransformerTokens = config.transformerMaxTokens();
        this.maxDocumentTokens = Math.min(config.maxDocumentTokens(), maxTransformerTokens);
        this.maxQueryTokens = Math.min(config.maxQueryTokens(), maxTransformerTokens);
        this.startSequenceToken = config.transformerStartSequenceToken();
        this.endSequenceToken = config.transformerEndSequenceToken();
        this.maskSequenceToken = config.transformerMaskToken();
        this.padSequenceToken = config.transformerPadToken();
        this.querySequenceToken = config.queryTokenId();
        this.documentSequenceToken = config.documentTokenId();
        this.poolFactor = config.poolFactor();
        this.prependQuery = config.prependQuery();
        this.prependDocument = config.prependDocument();

        Path tokenizerPath = config.tokenizerPath();
        var builder = new HuggingFaceTokenizer.Builder()
                .addSpecialTokens(false)
                .addDefaultModel(tokenizerPath)
                .setPadding(false);
        var info = HuggingFaceTokenizer.getModelInfo(tokenizerPath);
        if (info.maxLength() == -1 || info.truncation() != LONGEST_FIRST) {
            int maxLength = info.maxLength() > 0 && info.maxLength() <= maxTransformerTokens
                    ? info.maxLength()
                    : maxTransformerTokens;
            builder.setTruncation(true).setMaxLength(maxLength);
        }
        this.tokenizer = builder.build();

        this.skipTokens = new HashSet<>();
        PUNCTUATION.chars().forEach(c ->
                this.skipTokens.addAll(tokenizer.encode(Character.toString((char) c), null).ids()));

        var onnxOpts = OnnxEvaluatorOptions.createDefault();
        this.evaluator = onnx.evaluatorOf(config.transformerModelPath().toString(), onnxOpts);
        validateModel();
    }

    private void validateModel() {
        Map<String, TensorType> inputs = evaluator.getInputInfo();
        validateName(inputs, inputIdsName, "input");
        validateName(inputs, attentionMaskName, "input");
        Map<String, TensorType> outputs = evaluator.getOutputInfo();
        validateName(outputs, outputName, "output");
    }

    private static void validateName(Map<String, TensorType> types, String name, String type) {
        if (!types.containsKey(name)) {
            throw new IllegalArgumentException("Model does not contain required " + type + ": '" + name + "'. " +
                    "Model contains: " + String.join(",", types.keySet()));
        }
    }

    @Override
    public List<Integer> embed(String text, Context context) {
        throw new UnsupportedOperationException("This embedder only supports embed with tensor type");
    }

    @Override
    public Tensor embed(String text, Context context, TensorType tensorType) {
        if (!validTensorType(tensorType))
            throw new IllegalArgumentException("Invalid tensor target. Expected a mixed 2-d mapped-indexed tensor, got " + tensorType);
        text = prependInstruction(text, context);
        if (context.getDestinationType() == Context.DestinationType.QUERY)
            return embedQuery(text, context, tensorType);
        else
            return embedDocument(text, context, tensorType);
    }

    String prependInstruction(String text, Context context) {
        if (prependQuery != null && !prependQuery.isEmpty()
                && context.getDestinationType() == Context.DestinationType.QUERY) {
            return prependQuery + text;
        }
        if (prependDocument != null && !prependDocument.isEmpty()) {
            return prependDocument + text;
        }
        return text;
    }

    @Override
    public void deconstruct() {
        evaluator.close();
        tokenizer.close();
    }

    // ----- Query embedding (no pooling, pad with MASK) -----

    private Tensor embedQuery(String text, Context context, TensorType tensorType) {
        if (tensorType.valueType() == TensorType.Value.INT8)
            throw new IllegalArgumentException("ColBERT query embedding does not support int8 tensor type");

        EmbeddingResult result = lookupOrEvaluate(context, text, true);
        var output = (IndexedTensor) result.outputs.get(outputName);
        return toFloatTensor(output, tensorType, result.inputIdSize);
    }

    // ----- Document embedding (with optional pooling) -----

    private Tensor embedDocument(String text, Context context, TensorType tensorType) {
        EmbeddingResult result = lookupOrEvaluate(context, text, false);
        var modelOutput = (IndexedTensor) result.outputs.get(outputName);

        if (poolFactor <= 1) {
            // No pooling – behave like standard ColBertEmbedder
            if (tensorType.valueType() == TensorType.Value.INT8)
                return toBitTensor(modelOutput, tensorType, result.inputIdSize);
            else
                return toFloatTensor(modelOutput, tensorType, result.inputIdSize);
        }

        // Extract float embeddings from model output
        int nTokens = result.inputIdSize;
        int dim = (int) modelOutput.shape()[2];
        double[] embeddings = new double[nTokens * dim];
        for (int t = 0; t < nTokens; t++) {
            for (int d = 0; d < dim; d++) {
                embeddings[t * dim + d] = modelOutput.get(0, t, d);
            }
        }

        // Apply hierarchical token pooling
        double[] pooled = HierarchicalTokenPooling.poolTokens(embeddings, nTokens, dim, poolFactor, true);
        int nPooled = pooled.length / dim;

        // Build output tensor
        if (tensorType.valueType() == TensorType.Value.INT8) {
            return toBitTensorFromPooled(pooled, nPooled, dim, tensorType);
        } else {
            return toFloatTensorFromPooled(pooled, nPooled, dim, tensorType);
        }
    }

    // ----- Model inference -----

    TransformerInput buildTransformerInput(List<Long> tokens, int maxTokens, boolean isQuery) {
        if (!isQuery)
            tokens = tokens.stream().filter(t -> !skipTokens.contains(t)).toList();

        List<Long> inputIds = new ArrayList<>(maxTokens);
        List<Long> attentionMask = new ArrayList<>(maxTokens);

        if (tokens.size() > maxTokens - 3)
            tokens = tokens.subList(0, maxTokens - 3);

        inputIds.add(startSequenceToken);
        inputIds.add(isQuery ? querySequenceToken : documentSequenceToken);
        inputIds.addAll(tokens);
        inputIds.add(endSequenceToken);

        int inputLength = inputIds.size();
        long padTokenId = isQuery ? maskSequenceToken : padSequenceToken;
        int padding = isQuery ? maxTokens - inputLength : 0;

        for (int i = 0; i < padding; i++) inputIds.add(padTokenId);
        for (int i = 0; i < inputLength; i++) attentionMask.add(1L);
        for (int i = 0; i < padding; i++) attentionMask.add(0L);

        return new TransformerInput(inputIds, attentionMask);
    }

    EmbeddingResult lookupOrEvaluate(Context context, String text, boolean isQuery) {
        var key = new EmbedderCacheKey(context.getEmbedderId(), text);
        return context.computeCachedValueIfAbsent(key, () -> evaluate(context, text, isQuery));
    }

    private EmbeddingResult evaluate(Context context, String text, boolean isQuery) {
        var start = System.nanoTime();
        var encoding = tokenizer.encode(text, context.getLanguage());
        runtime.sampleSequenceLength(encoding.ids().size(), context);

        TransformerInput input = buildTransformerInput(
                encoding.ids(), isQuery ? maxQueryTokens : maxDocumentTokens, isQuery);

        Tensor inputIdsTensor = createTensorRepresentation(input.inputIds, "d1");
        Tensor attentionMaskTensor = createTensorRepresentation(input.attentionMask, "d1");

        Map<String, Tensor> outputs = evaluator.evaluate(Map.of(
                inputIdsName, inputIdsTensor.expand("d0"),
                attentionMaskName, attentionMaskTensor.expand("d0")));

        runtime.sampleEmbeddingLatency((System.nanoTime() - start) / 1_000_000d, context);
        return new EmbeddingResult(input.inputIds.size(), outputs);
    }

    // ----- Tensor construction -----

    static Tensor toFloatTensor(IndexedTensor result, TensorType type, int nTokens) {
        int wantedDim = type.indexedSubtype().dimensions().get(0).size().get().intValue();
        int resultDim = (int) result.shape()[2];
        if (wantedDim > resultDim)
            throw new IllegalArgumentException("Cannot map " + resultDim + " dims into " + wantedDim);

        Tensor.Builder builder = Tensor.Builder.of(type);
        for (int t = 0; t < nTokens; t++)
            for (int d = 0; d < wantedDim; d++)
                builder.cell(TensorAddress.of(t, d), result.get(0, t, d));
        return builder.build();
    }

    static Tensor toBitTensor(IndexedTensor result, TensorType type, int nTokens) {
        int wantedDim = type.indexedSubtype().dimensions().get(0).size().get().intValue();
        int floatDim = 8 * wantedDim;
        if (floatDim > (int) result.shape()[2])
            throw new IllegalArgumentException("Cannot pack " + result.shape()[2] + " dims into " + wantedDim);

        Tensor.Builder builder = Tensor.Builder.of(type);
        for (int t = 0; t < nTokens; t++) {
            BitSet bits = new BitSet(8);
            int key = 0;
            for (int d = 0; d < floatDim; d++) {
                int bitIndex = 7 - (d % 8);
                if (result.get(0, t, d) > 0.0) bits.set(bitIndex); else bits.clear(bitIndex);
                if ((d + 1) % 8 == 0) {
                    byte[] bytes = bits.toByteArray();
                    builder.cell(TensorAddress.of(t, key), bytes.length == 0 ? 0 : bytes[0]);
                    key++;
                    bits = new BitSet(8);
                }
            }
        }
        return builder.build();
    }

    /** Build a float tensor from pooled double[] embeddings. */
    private static Tensor toFloatTensorFromPooled(double[] pooled, int nPooled, int dim, TensorType type) {
        int wantedDim = type.indexedSubtype().dimensions().get(0).size().get().intValue();
        if (wantedDim > dim)
            throw new IllegalArgumentException("Cannot map " + dim + " dims into " + wantedDim);

        Tensor.Builder builder = Tensor.Builder.of(type);
        for (int t = 0; t < nPooled; t++)
            for (int d = 0; d < wantedDim; d++)
                builder.cell(TensorAddress.of(t, d), pooled[t * dim + d]);
        return builder.build();
    }

    /** Build an int8 bit-packed tensor from pooled double[] embeddings. */
    private static Tensor toBitTensorFromPooled(double[] pooled, int nPooled, int dim, TensorType type) {
        int wantedDim = type.indexedSubtype().dimensions().get(0).size().get().intValue();
        int floatDim = 8 * wantedDim;
        if (floatDim > dim)
            throw new IllegalArgumentException("Cannot pack " + dim + " dims into " + wantedDim);

        Tensor.Builder builder = Tensor.Builder.of(type);
        for (int t = 0; t < nPooled; t++) {
            BitSet bits = new BitSet(8);
            int key = 0;
            int tOff = t * dim;
            for (int d = 0; d < floatDim; d++) {
                int bitIndex = 7 - (d % 8);
                if (pooled[tOff + d] > 0.0) bits.set(bitIndex); else bits.clear(bitIndex);
                if ((d + 1) % 8 == 0) {
                    byte[] bytes = bits.toByteArray();
                    builder.cell(TensorAddress.of(t, key), bytes.length == 0 ? 0 : bytes[0]);
                    key++;
                    bits = new BitSet(8);
                }
            }
        }
        return builder.build();
    }

    private boolean validTensorType(TensorType target) {
        return target.dimensions().size() == 2 && target.indexedSubtype().rank() == 1;
    }

    private IndexedTensor createTensorRepresentation(List<Long> input, String dimension) {
        int size = input.size();
        TensorType type = new TensorType.Builder(TensorType.Value.FLOAT).indexed(dimension, size).build();
        IndexedTensor.Builder builder = IndexedTensor.Builder.of(type);
        for (int i = 0; i < size; i++) builder.cell(input.get(i), i);
        return builder.build();
    }
}
