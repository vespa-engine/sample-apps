package ai.vespa.searcher;

import ai.vespa.tokenizer.BertModelConfig;
import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.component.chain.Chain;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import org.junit.Test;
import com.yahoo.search.Query;

import java.io.IOException;
import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class QASearcherTest {
    static BertModelConfig bertModelConfig;

    static {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(new com.yahoo.config.FileReference("src/test/resources/bert-base-uncased-vocab.txt")).max_input(128);
        bertModelConfig = builder.build();
    }

    @Test
    public void testSearcher() throws IOException  {
        Query query = new Query("/search/?query=Who+landed+on+the+moon");
        assertEquals("Does not match", "Who landed on the moon", query.getModel().getQueryString());
        Result result = execute(query, new QASearcher(new BertTokenizer(bertModelConfig)));
        Optional<Tensor> question_token_ids = query.getRanking().getFeatures().getTensor("query(query_token_ids)");
        assertTrue("Tensor was empty", !question_token_ids.isEmpty());
        Tensor token_ids = question_token_ids.get();
        assertEquals(bertModelConfig.max_input(),token_ids.size());
    }


    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }
}
