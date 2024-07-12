// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.reranker;

import com.yahoo.component.annotation.Inject;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.yolean.Exceptions;
import com.yahoo.yolean.chain.After;

import java.io.IOException;
import java.util.Map;

/**
 * A searcher which federates to another Vespa application.
 *
 * @author bratseth
 */
@After("*")
public class VespaSearcher extends Searcher {

    private final VespaClient client;
    private final ResultReader resultReader = new ResultReader();

    @Inject
    public VespaSearcher(RerankerConfig config) {
        this(new VespaClient(config.endpoint()));
    }

    public VespaSearcher(VespaClient client) {
        this.client = client;
    }

    @Override
    public Result search(Query query, Execution execution) {
        try {
            if (query.properties().getBoolean("metrics.ignore")) return execution.search(query); // Don't pass on warmup
            VespaClient.Response response = client.search(query.getHttpRequest(),
                                                          Map.of("ranking", query.getRanking().getProfile()));
            Result result = execution.search(query);
            resultReader.read(response.responseBody(), result);
            if (response.statusCode() != 200 && result.hits().getError() == null)
                result.hits().addError(ErrorMessage.createBackendCommunicationError("Backend returned status " +
                                                                                    response.statusCode() +
                                                                                    ": " + response.responseBody()));
            return result;
        }
        catch (IOException e) {
            return new Result(query, ErrorMessage.createBackendCommunicationError(Exceptions.toMessageString(e)));
        }
    }

}
