// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.component.chain.Chain;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.yql.MinimalQueryInserter;
import org.junit.jupiter.api.Test;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static java.net.URLEncoder.encode;

public class EquivSearcherTest {

    List<String> queries = List.of(
            "select * from sources * where artist contains \"metallica\"",
            "select * from sources * where artist contains \"metallica\" AND album contains \"puppets\"");

    @Test
    void testMetallica() {
        Chain<Searcher> myChain = new Chain<>(new MinimalQueryInserter(), new EquivSearcher());  // added to chain in this order
        Execution.Context context = Execution.Context.createContextStub();
        Execution execution = new Execution(myChain, context);

        for (String yql: queries) {
            Query query = new Query("/search/?yql=" + encode(yql, StandardCharsets.UTF_8));
            query.getTrace().setLevel(6);
            Result result = execution.search(query);
            System.out.println(result.getContext(false).getTrace());
        }
    }

}
