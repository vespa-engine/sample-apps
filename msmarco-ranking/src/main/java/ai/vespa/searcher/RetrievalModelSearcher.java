// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

import ai.vespa.tokenizer.BertTokenizer;
import com.google.inject.Inject;
import com.yahoo.language.Language;
import com.yahoo.language.Linguistics;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.prelude.query.WeakAndItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;

import java.util.ArrayList;
import java.util.List;

public class RetrievalModelSearcher extends Searcher {

    private static final String QUERY_TENSOR_NAME = "query(query_token_ids)";

    private final Linguistics linguistics;
    private final BertTokenizer tokenizer;
    private final TensorType questionInputTensorType = TensorType.fromSpec("tensor<float>(d0[32])");

    public enum RetrievalMethod {
        SPARSE,
        DENSE,
        RANK
    }

    @Inject
    public RetrievalModelSearcher(Linguistics linguistics, BertTokenizer tokenizer) {
        this.linguistics = linguistics;
        this.tokenizer = tokenizer;
    }

    @Override
    public Result search(Query query, Execution execution) {
        String queryInput = query.getModel().getQueryString();
        if (query.getModel().getQueryString() == null || query.getModel().getQueryString().length() == 0)
            return new Result(query, ErrorMessage.createBadRequest("No query input"));

        Tensor questionTokenIds = getQueryTokenIds(queryInput, questionInputTensorType.sizeOfDimension("d0").get().intValue());
        query.getRanking().getFeatures().put(QUERY_TENSOR_NAME, questionTokenIds);

        query.getRanking().setRerankCount(query.properties().getInteger("phase.count", 24));

        switch(getMethod(query))  {
            case SPARSE:
                String wandField = query.properties().getString("wand.field", "default");
                int wandHits = query.properties().getInteger("wand.hits", query.getHits());
                query.getModel().getQueryTree().setRoot(sparseRetrieval(queryInput, wandField, wandHits));
                break;
            case DENSE:
                String annField = query.properties().getString("ann.field", "mini_document_embedding");
                int annHits = query.properties().getInteger("ann.hits", query.getHits());
                int annExtraHits = query.properties().getInteger("ann.extra-hits", query.getHits());
                String queryTensorName = query.properties().getString("ann.query", "query_embedding");
                query.getModel().getQueryTree().setRoot(
                        denseRetrieval(annHits, annExtraHits, annField, queryTensorName,
                                       query.properties().getBoolean("ann.brute-force")));
                break;
        }
        return execution.search(query);
    }

    private List<String> tokenize(String query) {
        Iterable<Token> tokens = this.linguistics.getTokenizer().tokenize(query, Language.ENGLISH, StemMode.NONE, true);
        List<String> queryTokens = new ArrayList<>();
        for (Token t : tokens) {
            if (t.isIndexable())
                queryTokens.add(t.getTokenString());
        }
        return queryTokens;
    }

    private WeakAndItem sparseRetrieval(String queryInput, String field, int hits) {
        WeakAndItem wand = new WeakAndItem();
        wand.setN(hits);
        for (String t : tokenize(queryInput)) {
            wand.addItem(new WordItem(t, field, true));
        }
        return wand;
    }

    private NearestNeighborItem denseRetrieval(int annHits, int annExtraHits, String field, String queryTensorName, boolean approximate) {
        NearestNeighborItem nn = new NearestNeighborItem(field, queryTensorName);
        nn.setAllowApproximate(!approximate);
        nn.setTargetNumHits(annHits);
        nn.setHnswExploreAdditionalHits(annExtraHits);
        return nn;
    }

    public static RetrievalMethod getMethod(Query query) {
        String method = query.properties().getString("retriever", "sparse");
        switch (method) {
            case "sparse": return RetrievalMethod.SPARSE;
            case "dense": return RetrievalMethod.DENSE;
            case "rank": return RetrievalMethod.RANK;
            default: return RetrievalMethod.SPARSE;
        }
    }

    protected static boolean needQueryEmbedding(Query query) {
        RetrievalMethod method = getMethod(query);
        return (method == RetrievalMethod.DENSE || method == RetrievalMethod.RANK);
    }

    private Tensor getQueryTokenIds(String queryInput, int maxLength) {
        List<Integer> tokensIds = tokenizer.tokenize(queryInput, maxLength, true);
        Tensor.Builder builder = Tensor.Builder.of(questionInputTensorType);
        int i = 0;
        for (Integer tokenId : tokensIds)
            builder.cell(tokenId, i++);
        return builder.build();
    }

}
