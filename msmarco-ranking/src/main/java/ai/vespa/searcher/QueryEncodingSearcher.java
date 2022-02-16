// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.AsyncExecution;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.searchchain.FutureResult;
import com.yahoo.search.searchchain.SearchChain;
import com.yahoo.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/**
 * Searcher that asynchronously invokes the colbert and embedding search chains
 * to compute the colbert query tensor multi-representation and the single dense query embedding representation.
 *
 * The requests never reaches the backend, but we reuse the request response paradigm to return
 * the result of the computation as hits.
 *
 */

public class QueryEncodingSearcher  extends Searcher {

    /**
     * Tensor names which are passed to the content cluster and
     * which types needs to be known in configuration
     * See src/main/application/search/query-profiles/types/root.xml for
     * tensor type definition
     */
    private final String colbertTensorName = "query(qt)";
    private final String embeddingTensorName = "query(query_embedding)";

    @Override
    public Result search(Query query, Execution execution) {
        SearchChain colbert = execution.searchChainRegistry().getComponent("colbert");
        SearchChain query_embedding = execution.searchChainRegistry().getComponent("embedding");
        List<SearchChain> targets = new ArrayList<>();
        if (RetrievalModelSearcher.needQueryEmbedding(query))
            targets.add(query_embedding);
        targets.add(colbert);
        federateAndUpdateQuery(query, execution, targets);
        return execution.search(query);
    }

    private void federateAndUpdateQuery(Query query, Execution execution, List<SearchChain> chains) {
        List<FutureResult> results = new ArrayList<>();
        for(SearchChain c: chains) {
            Execution subExecution = new Execution(c, execution.context());
            results.add(new AsyncExecution(subExecution).search(query.clone()));
        }
        for (FutureResult f : results) {
            Result r = f.get();
            if(r.getTotalHitCount() == 0)
                throw new RuntimeException("Unexpected 0 hits from query encoder, this is a server error.");
            Hit hit = r.hits().get(0);
            Tensor tensor = (Tensor)hit.getField("tensor");
            if (hit.getSource().equals("colbert")) {
                query.getRanking().getFeatures().put(colbertTensorName, tensor);
            }
            else if(hit.getSource().equals("embedding")) {
                query.getRanking().getFeatures().put(embeddingTensorName, tensor);
            }
        }
    }

}
