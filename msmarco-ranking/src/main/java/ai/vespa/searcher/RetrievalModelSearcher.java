// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

import com.google.inject.Inject;
import com.yahoo.language.Language;
import com.yahoo.language.Linguistics;
import com.yahoo.language.process.Embedder;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.language.wordpiece.WordPieceEmbedder;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.prelude.query.WeakAndItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

public class RetrievalModelSearcher extends Searcher {

    private static final String QUERY_TENSOR_NAME = "query(query_token_ids)";
    private static final int MAX_QUERY_LENGTH = 32;

    private final Linguistics linguistics;
    private WordPieceEmbedder embedder;

    public enum RetrievalMethod {
        SPARSE,
        DENSE,
        RANK
    }

    @Inject
    public RetrievalModelSearcher(Linguistics linguistics, WordPieceEmbedder embedder) {
        this.linguistics = linguistics;
        this.embedder = embedder;
    }

    @Override
    public Result search(Query query, Execution execution) {
        String queryInput = query.getModel().getQueryString();
        if (query.getModel().getQueryString() == null || query.getModel().getQueryString().length() == 0)
            return new Result(query, ErrorMessage.createBadRequest("No query input"));
        List<Integer> bertTokenIds = this.embedder.embed(queryInput, new Embedder.Context("q"));
        if(bertTokenIds.size() > MAX_QUERY_LENGTH)
            bertTokenIds = bertTokenIds.subList(0,MAX_QUERY_LENGTH);

        QueryTensorInput queryTensorInput = new QueryTensorInput(bertTokenIds);
        QueryTensorInput.setTo(query.properties(),queryTensorInput);
        Tensor queryTensor = queryTensorInput.getTensorRepresentation(
                queryTensorInput.getQueryTokenIdsPadded(MAX_QUERY_LENGTH,0),"d0");
        query.getRanking().getFeatures().put(QUERY_TENSOR_NAME, queryTensor);
        query.getRanking().setRerankCount(query.properties().getInteger("rerank-count", 1000));

        switch(getMethod(query))  {
            case SPARSE:
                String wandField = query.properties().getString("wand.field", "default");
                int wandHits = query.properties().getInteger("wand.hits", query.getHits());
                query.getModel().getQueryTree().setRoot(sparseRetrieval(queryInput,
                        query.getModel().getLanguage(),wandField, wandHits));
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

    /**
     * Tokenize the input queryString. Stemming is not performed as that is handled by downstream
     * content cluster searcher (StemmingSearcher).
     *
     * @param queryString the input query string
     * @param language the language to use when tokenization
     * @return List of strings
     */

    private List<String> tokenize(String queryString, Language language) {
        Iterable<Token> tokens = this.linguistics.getTokenizer().tokenize(queryString, language, StemMode.NONE, true);
        List<String> queryTokens = new ArrayList<>();
        for (Token t : tokens) {
            if (t.isIndexable())
                queryTokens.add(t.getTokenString());
        }
        return queryTokens;
    }

    /**
     * Create a WeakAnd query from the query representation
     * @param queryInput The string query
     * @param field The field to run weakAnd over, can be both a regular field and a fieldset
     * @param language from the query
     * @param hits The target hits
     * @return The WeakAndItem
     */

    private WeakAndItem sparseRetrieval(String queryInput,  Language language, String field, int hits) {
        WeakAndItem wand = new WeakAndItem();
        wand.setN(hits);
        for (String t : tokenize(queryInput,language)) {
            //Note that isFromQuery=true allows the term to be stemmed by StemmingSearcher invoked in the
            //content cluster specific chain. Use &tracelevel=3 to trace requests
            wand.addItem(new WordItem(t, field, true));
        }
        return wand;
    }

    /**
     * Create the NN query operator
     * @param annHits target number of hits to return to ranking phases
     * @param annExtraHits extra hits to improve ANN recall as compared with brute force NN
     * @param field The dense vector field to search with the NN
     * @param queryTensorName The name of the dense query tensor
     * @param approximate Allow approximate search
     * @return The NN operator
     */

    private NearestNeighborItem denseRetrieval(int annHits, int annExtraHits, String field, String queryTensorName, boolean approximate) {
        NearestNeighborItem nn = new NearestNeighborItem(field, queryTensorName);
        nn.setAllowApproximate(!approximate);
        nn.setTargetNumHits(annHits);
        nn.setHnswExploreAdditionalHits(annExtraHits);
        return nn;
    }

    /**
     * Determine what retriever to be be used
     * @param query query to read properties from
     * @return The Retrieval Method
     */
    private static RetrievalMethod getMethod(Query query) {
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
}
