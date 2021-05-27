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
 * Searcher which asynchronously invokes the colbert and embedding search chains
 * to compute the colbert query tensor representation and the dense query embedding representation
 *
 */

public class QueryEncodingSearcher  extends Searcher {

    private String colbertTensorName = "query(qt)";
    private String embeddingTensorName = "query(query_embedding)";

    @Override
    public Result search(Query query, Execution execution) {
        SearchChain colbert = execution.searchChainRegistry().getComponent("colbert");
        SearchChain query_embedding = execution.searchChainRegistry().getComponent("embedding");
        List<SearchChain> targets = new ArrayList<>();
        if(RetrievalModelSearcher.needQueryEmbedding(query))
            targets.add(query_embedding);
        targets.add(colbert);
        federateAndUpdateQuery(query,execution,targets);
        return execution.search(query);
    }

    private void federateAndUpdateQuery(Query query, Execution execution, List<SearchChain> chains) {
        List<FutureResult> results = new ArrayList<>();
        for(SearchChain c: chains) {
            Execution subExecution = new Execution(c, execution.context());
            results.add(new AsyncExecution(subExecution).search(query.clone()));
        }
        for (FutureResult f: results) {
            Result r = f.get();
            Hit hit = r.hits().get(0);
            Tensor tensor = (Tensor)hit.getField("tensor");
            if(hit.getSource().equals("colbert")) {
                query.getRanking().getFeatures().put(colbertTensorName, tensor);
                query.trace("colbert tensor " + tensor,3);
            }
            else if(hit.getSource().equals("embedding")) {
                query.getRanking().getFeatures().put(embeddingTensorName, tensor);
                query.trace("embedding tensor " + tensor,3);
            }
        }
    }
}
