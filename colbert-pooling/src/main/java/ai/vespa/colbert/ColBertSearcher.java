// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.colbert;

import com.yahoo.component.annotation.Inject;
import com.yahoo.component.provider.ComponentRegistry;
import com.yahoo.language.process.Embedder;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.prelude.query.OrItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;

import java.util.BitSet;

/**
 * Searcher that embeds the user query with ColBERT, binarises the per-token
 * embeddings, and rewrites the query to use {@code nearestNeighbor} operators
 * on the binary HNSW-indexed field.
 * <p>
 * Also sets the float multi-vector query tensor {@code qt} for the second-phase
 * exact MaxSim reranking, and the binary multi-vector {@code qtb} for the
 * first-phase hamming MaxSim scoring.
 * <p>
 * Just send text:
 * <pre>
 *   /search/?query=machine+learning
 * </pre>
 */
public class ColBertSearcher extends Searcher {

    private static final String BINARY_FIELD = "colbert_pooled_binary";
    private static final int TARGET_HITS_PER_TOKEN = 10;
    private static final int MAX_QUERY_TOKENS = 32;

    private static final TensorType FLOAT_QUERY_TYPE =
            TensorType.fromSpec("tensor<float>(qt{},x[128])");
    private static final TensorType BINARY_QUERY_TYPE =
            TensorType.fromSpec("tensor<int8>(qt{},x[16])");
    private static final TensorType SINGLE_BINARY_TYPE =
            TensorType.fromSpec("tensor<int8>(x[16])");

    private final Embedder embedder;

    @Inject
    public ColBertSearcher(ComponentRegistry<Embedder> embedders) {
        this.embedder = embedders.getComponent("colbert");
    }

    @Override
    public Result search(Query query, Execution execution) {
        String queryText = query.getModel().getQueryString();
        if (queryText == null || queryText.isBlank()) {
            return execution.search(query);
        }

        // Skip if query tensors are already set (client-side embedding)
        if (query.getRanking().getFeatures().getTensor("query(qt)").isPresent()) {
            return execution.search(query);
        }

        // 1. Embed query → float multi-vector
        var context = new Embedder.Context("query(qt)");
        Tensor floatTensor = embedder.embed(queryText, context, FLOAT_QUERY_TYPE);
        query.getRanking().getFeatures().put("query(qt)", floatTensor);

        // 2. Count tokens
        int nTokens = 0;
        for (var iter = floatTensor.cellIterator(); iter.hasNext(); ) {
            var cell = iter.next();
            int mapped = (int) cell.getKey().numericLabel(0);
            nTokens = Math.max(nTokens, mapped + 1);
        }
        nTokens = Math.min(nTokens, MAX_QUERY_TOKENS);
        int dim = 128;
        int packedDim = dim / 8;

        // 3. Binarise tokens → qtb (multi-vector) + rqN (single-vector per token)
        Tensor.Builder qtbBuilder = Tensor.Builder.of(BINARY_QUERY_TYPE);
        Tensor[] rqTensors = new Tensor[nTokens];

        for (int t = 0; t < nTokens; t++) {
            Tensor.Builder rqBuilder = Tensor.Builder.of(SINGLE_BINARY_TYPE);
            BitSet bits = new BitSet(8);
            int key = 0;
            for (int d = 0; d < dim; d++) {
                int bitIndex = 7 - (d % 8);
                double value = floatTensor.get(TensorAddress.of(t, d));
                if (value > 0.0) bits.set(bitIndex);
                else bits.clear(bitIndex);

                if ((d + 1) % 8 == 0) {
                    byte[] bytes = bits.toByteArray();
                    byte packed = bytes.length == 0 ? 0 : bytes[0];
                    qtbBuilder.cell(TensorAddress.of(t, key), packed);
                    rqBuilder.cell(TensorAddress.of(key), packed);
                    key++;
                    bits = new BitSet(8);
                }
            }
            rqTensors[t] = rqBuilder.build();
        }
        query.getRanking().getFeatures().put("query(qtb)", qtbBuilder.build());

        // 4. Build nearestNeighbor OR query
        OrItem nnOr = new OrItem();
        for (int t = 0; t < nTokens; t++) {
            String rqName = "rq" + t;
            query.getRanking().getFeatures().put("query(" + rqName + ")", rqTensors[t]);

            NearestNeighborItem nn = new NearestNeighborItem(BINARY_FIELD, rqName);
            nn.setTargetHits(TARGET_HITS_PER_TOKEN);
            nn.setAllowApproximate(true);
            nnOr.addItem(nn);
        }

        // 5. Set as query root (nearestNeighbor is the sole retrieval mechanism)
        query.getModel().getQueryTree().setRoot(nnOr);

        return execution.search(query);
    }
}
