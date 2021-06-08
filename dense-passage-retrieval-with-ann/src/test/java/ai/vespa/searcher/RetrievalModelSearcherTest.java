// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;

import ai.vespa.tokenizer.BertModelConfig;
import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.component.chain.Chain;
import com.yahoo.data.access.slime.SlimeAdapter;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.Query;
import com.yahoo.slime.Cursor;
import com.yahoo.slime.Slime;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.serialization.TypedBinaryFormat;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;


public class RetrievalModelSearcherTest {

    static BertModelConfig bertModelConfig;
    static BertTokenizer tokenizer;

    static {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(new com.yahoo.config.FileReference("src/test/resources/bert-base-uncased-vocab.txt")).max_input(128);
        bertModelConfig = builder.build();
        try {
            tokenizer = new BertTokenizer(bertModelConfig, new SimpleLinguistics());
        } catch (IOException e) {
            fail("Caught IO expcetion while loading bert model config");
        }
    }


    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }

    @Test
    public void test_searcher(){
        RetrievalModelSearcher retrievalModelSearcher = new RetrievalModelSearcher(tokenizer);
        MockBackend mock = new MockBackend();
        Query query = new Query("/search/?query=hello+nelson");

        Result r = execute(query,retrievalModelSearcher,mock);
        Optional<Tensor> t = query.getRanking().getFeatures().getTensor("query(query_embedding)");
        assertFalse(t.isEmpty());
        Tensor queryEmbedding = t.get();
        assertEquals(768, queryEmbedding.size());

        t = query.getRanking().getFeatures().getTensor("query(query_hash)");
        assertFalse(t.isEmpty(),"No query hash tensor");
        Tensor hashEmbedding = t.get();
        assertEquals(96, hashEmbedding.size());

        NearestNeighborItem nn =(NearestNeighborItem)query.getModel().getQueryTree().getRoot();
        assertEquals("hash",nn.getIndexName());
        assertEquals("query_hash",nn.getQueryTensorName());

    }


    private static class MockBackend extends Searcher {
        @Override
        public Result search(Query query, Execution execution) {
            if(isEncoderQuery(query)) {
                Result result = execution.search(query);
                result.setTotalHitCount(1);
                Hit hit = new Hit("query",1.0);
                hit.setField("summaryfeatures",getFeatureData("rankingExpression(cls_embedding)"));
                result.hits().add(hit);
                return result;
            }
            return execution.search(query);
        }
        private boolean isEncoderQuery(Query query) {
            return query.getModel().getRestrict().contains("query");
        }
    }

    private static FeatureData getFeatureData(String name) {
        Cursor features = new Slime().setObject();

        Tensor.Builder b = Tensor.Builder.of(TensorType.fromSpec("tensor<float>(d2[768])"));
        for (long i = 0; i < 768; i++)
            b.cell(TensorAddress.of(i),1.0);
        Tensor embedding = b.build();
        features.setData(name, TypedBinaryFormat.encode(embedding));
        return new FeatureData(new SlimeAdapter(features));
    }

}


