// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher.colbert;

import ai.vespa.colbert.ColbertConfig;
import ai.vespa.tokenizer.BertModelConfig;
import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.component.chain.Chain;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.query.ranking.RankFeatures;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.Query;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.data.access.slime.SlimeAdapter;
import com.yahoo.slime.Cursor;
import com.yahoo.slime.Slime;
import com.yahoo.tensor.serialization.TypedBinaryFormat;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class ColbertSearcherTest {

    private static final ColbertConfig colbertConfig;
    private static BertTokenizer tokenizer;

    private static final int DIM = 32;
    private static final int MAX_QUERY_LENGHT = 32;
    private static final String outputName = "onnxModel(encoder).contextual";
    private static final String profile = "query_encoder";

    static {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(new com.yahoo.config.FileReference("src/test/resources/bert-base-uncased-vocab.txt"))
               .max_input(128);
        BertModelConfig bertModelConfig = builder.build();
        try {
            tokenizer = new BertTokenizer(bertModelConfig, new SimpleLinguistics());
        } catch (IOException e) {
            fail("IO Error during bert model read");
        }

        ColbertConfig.Builder colbertBuilder = new ColbertConfig.Builder();
        colbertBuilder.dim(DIM).max_query_length(MAX_QUERY_LENGHT).output_name(outputName).rank_profile(profile);
        colbertConfig = colbertBuilder.build();
    }

    @Test
    public void testColBERTSearcher() {
        ColBERTSearcher colBERTSearcher = new ColBERTSearcher(tokenizer,colbertConfig);
        MockBackend backend = new MockBackend();
        Query query = new Query("?query=what+was+the+impact+of+the+manhattan+project");
        Result result = execute(query, colBERTSearcher,backend);
        assertEquals(1, result.getConcreteHitCount());
        Hit tensorHit = result.hits().get(0);
        assertEquals("colbert",tensorHit.getSource());
        Tensor colBertTensor = (Tensor)tensorHit.getField("tensor");
        assertNotNull(colBertTensor);
        assertEquals(32*32, colBertTensor.size()); //32 query terms x 32 dim
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
                Hit hit = new Hit("query", 1.0);
                hit.setField("summaryfeatures", getFeatureData(outputName));
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
        Cursor features = new Slime().setObject();
        Tensor.Builder builder = Tensor.Builder.of("tensor<float>(d0[1],d1[32],d2[32])");
        for (int i = 0; i < MAX_QUERY_LENGHT;i++)
            for (int j=0; j < DIM; j++)
                builder.cell(TensorAddress.of(0, i, j), 0.1);
        Tensor embedding = builder.build();
        features.setData(name, TypedBinaryFormat.encode(embedding));
        return new FeatureData(new SlimeAdapter(features));
    }

}


