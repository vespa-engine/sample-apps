package ai.vespa.searcher;

import ai.vespa.tokenizer.BertModelConfig;
import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.component.chain.Chain;
import com.yahoo.data.access.slime.SlimeAdapter;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.slime.Cursor;
import com.yahoo.slime.Slime;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.serialization.TypedBinaryFormat;
import org.junit.jupiter.api.Test;


import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assertions.assertNotNull;


public class QueryEmbeddingSearcherTest {

    private static BertTokenizer tokenizer;


    private static String outputName = "rankingExpression(mean_token_embedding)";
    private static Tensor.Builder backEndTensor = Tensor.Builder.of("tensor<float>(d2[384])");

    static {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(new com.yahoo.config.FileReference("src/test/resources/bert-base-uncased-vocab.txt")).max_input(512);
        BertModelConfig bertModelConfig = builder.build();
        try {
            tokenizer = new BertTokenizer(bertModelConfig, new SimpleLinguistics());
        } catch (IOException e) {
            fail("IO Error during bert model read");
        }
    }

    @Test
    public void testEmbeddingSearcher() {
        QueryEmbeddingSearcher embeddingSearcher = new QueryEmbeddingSearcher(tokenizer);
        MockBackend backend = new  MockBackend();
        Query query = new Query("?query=what+was+the+impact+of+the+manhattan+project");
        Result result = execute(query,embeddingSearcher,backend);
        assertEquals(1,result.getConcreteHitCount());
        Hit tensorHit = result.hits().get(0);
        assertEquals("embedding",tensorHit.getSource());
        Tensor embeddingTensor = (Tensor)tensorHit.getField("tensor");
        assertNotNull(embeddingTensor);
        assertEquals(384,embeddingTensor.size());
    }

    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }


    private static class MockBackend extends Searcher {

        @Override
        public Result search(Query query, Execution execution) {
            if (isEmbeddingQuery(query)) {
                Result result = execution.search(query);
                result.setTotalHitCount(1);
                Hit hit = new Hit("query",1.0);
                hit.setSource("query");
                hit.setField("summaryfeatures",getFeatureData(outputName));
                result.hits().add(hit);
                return result;
            } else {
                return execution.search(query);
            }
        }
        private boolean isEmbeddingQuery(Query query) {
            return (query.getModel().getRestrict().contains("query"));
        }
    }

    private static FeatureData getFeatureData(String name) {
        for(int i = 0; i < 384 ;i++ )
            backEndTensor.cell(TensorAddress.of(i),0.1);
        Tensor embedding = backEndTensor.build();
        Cursor features = new Slime().setObject();
        features.setData(name, TypedBinaryFormat.encode(embedding));
        return new FeatureData(new SlimeAdapter(features));
    }
}
