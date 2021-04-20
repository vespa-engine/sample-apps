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
import org.junit.Test;

import com.yahoo.data.access.slime.SlimeAdapter;
import com.yahoo.slime.Cursor;
import com.yahoo.slime.Slime;
import com.yahoo.tensor.serialization.TypedBinaryFormat;

import java.io.IOException;

import static org.junit.Assert.*;

public class ColbertSearcherTest {

    private static BertModelConfig bertModelConfig;
    private static ColbertConfig colbertConfig;
    private static BertTokenizer tokenizer;

    private static int DIM = 32;
    private static int MAX_QUERY_LENGHT = 32;
    private static String outputName = "onnxModel(encoder).contextual";
    private static String profile = "query_encoder";

    static {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(new com.yahoo.config.FileReference("src/test/resources/bert-base-uncased-vocab.txt")).max_input(128);
        bertModelConfig = builder.build();
        try {
            tokenizer = new BertTokenizer(bertModelConfig, new SimpleLinguistics());
        } catch (IOException e) {
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
        Result result = execute(query,colBERTSearcher,backend);
        RankFeatures features = result.getQuery().getRanking().getFeatures();
        Tensor colbertTensor = features.getTensor("query(qt)").get();
        assertNotNull(colbertTensor);
        assertEquals(32*32,colbertTensor.size()); //32 query terms x 32 dim
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
        Cursor features = new Slime().setObject();
        Tensor.Builder builder = Tensor.Builder.of("tensor<float>(d0[1],d1[32],d2[32])");
        for(int i = 0; i < 32;i++)
            for(int j=0; j < 32; j++)
                builder.cell(TensorAddress.of(0,i,j),0.1);
        Tensor embedding = builder.build();
        features.setData(name, TypedBinaryFormat.encode(embedding));
        return new FeatureData(new SlimeAdapter(features));

    }


}


