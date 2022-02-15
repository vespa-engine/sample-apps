// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

import ai.vespa.TokenizerFactory;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.component.chain.Chain;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.functions.Reduce;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONException;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;


public class ReRankingSearcherTest {

    private static class ReRankingSearcherTester {
        ReRankingSearcher searcher;
        Chain<Searcher> chain;

        private ReRankingSearcherTester(MockBackend backend){
            ModelsEvaluator eval = ModelsEvaluatorTester.create("src/main/application/models/");
            RetrievalModelSearcher retrievalModelSearcher = new RetrievalModelSearcher(
                    new SimpleLinguistics(), TokenizerFactory.getEmbedder());
            searcher = new ReRankingSearcher(eval);
            chain = new Chain<>(retrievalModelSearcher,searcher,backend);

        }
        private Result execute(Query query) {
            Execution execution = new Execution(chain, Execution.Context.createContextStub());
            return execution.search(query);
        }
    }


    @Test
    public void testReRankingSearcher() {
        MockBackend backend = new MockBackend("src/test/resources/sample_result.json");
        ReRankingSearcherTester tester = new ReRankingSearcherTester(backend);
        Query query = new Query("?query=what+was+the+Manhattan%20Project");
        Result result = tester.execute(query);
        assertEquals(10, result.hits().size());
        Hit best = result.hits().get(0);
        assertEquals(7, (int) best.getField("id"));
        for (Hit h : result.hits()) {
            String text = (String) h.getField("text");
            text = text.substring(0, 50).concat(" ...");
            System.out.format("Passage id=%d, score=%.2f, text=%s\n",
                    h.getField("id"), h.getRelevance().getScore(), text);
        }
    }

    @Test
    public void testModelInput() {
        MockBackend mock = new MockBackend("src/test/resources/sample_result.json");
        Result result = mock.search(new Query("?query=what+was+the+manhattan+project"), null);
        result.hits().trim(0,1);
        List<Integer> queryTokens = new ArrayList<>(Arrays.asList(2054, 2001, 1996, 7128, 262));
        ReRankingSearcher.BertModelBatchInput batchInput = ReRankingSearcher.buildModelInput(
                queryTokens,result,24,"text_token_ids");
        assertEquals(1,batchInput.inputIds.dimensionSizes().size(0));
        assertEquals(24,batchInput.inputIds.dimensionSizes().size(1));
        Tensor inputIds = batchInput.inputIds.reduce(Reduce.Aggregator.min,"d0");
        Tensor attentionMask = batchInput.attentionMask.reduce(Reduce.Aggregator.min,"d0");
        Tensor tokenTypeIds = batchInput.tokenTypeIds.reduce(Reduce.Aggregator.min,"d0");
        assertEquals("tensor<float>(d1[24]):[101.0, 2054.0, 2001.0, 1996.0, 7128.0, 262.0, 102.0,"
            + " 1996.0, 6522.0, 7069.0, 8654.0, 1012.0, 5594.0, 19841.0, 2860.0, 2003.0, 2003.0, 2615.0, 1011.0, 7378.0, 2061.0, 2115.0, 4684.0, 102.0]",
                inputIds.toString());
        assertEquals("tensor<float>(d1[24]):[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, " +
                "1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]", attentionMask.toString());

        assertEquals("tensor<float>(d1[24]):[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0," +
                " 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]",tokenTypeIds.toString());
    }


    @Test
    public void testModelInputTrim() {
        Result result = new Result(new Query());
        for(int i = 0; i < 2; i++) {
            TensorType type = new TensorType.Builder(TensorType.Value.FLOAT).
                    indexed("d0", i+1).build();
            IndexedTensor.Builder builder = IndexedTensor.Builder.of(type);
            for(int j = 0; j < i+1; j++)
                builder.cell(TensorAddress.of(j),99);
            Hit h = new Hit("test+"+i);
            h.setField("text_token_ids",builder.build());
            result.hits().add(h);
        }
        List<Integer> queryTokens = new ArrayList<>(Arrays.asList(66));
        ReRankingSearcher.BertModelBatchInput batchInput = ReRankingSearcher.buildModelInput(
                queryTokens,result,12,"text_token_ids");
        //Truncated to sequence length 6 instead of padded to max sequence length for improved inference
        assertEquals("tensor<float>(d0[2],d1[6]):[[101.0, 66.0, 102.0, 99.0, 102.0, 0.0], [101.0, 66.0, 102.0, 99.0, 99.0, 102.0]]",
                batchInput.inputIds.toString());

        queryTokens = new ArrayList<>(Arrays.asList(1,2,3,4,5,6));
        batchInput = ReRankingSearcher.buildModelInput(queryTokens,result,3,"text_token_ids");
        assertEquals("tensor<float>(d0[2],d1[3]):[[101.0, 102.0, 102.0], [101.0, 102.0, 102.0]]",batchInput.inputIds.toString());

        queryTokens = new ArrayList<>(Arrays.asList(1,2,3,4,5,6));
        batchInput = ReRankingSearcher.buildModelInput(queryTokens,result,2,"text_token_ids");
        assertEquals("tensor<float>(d0[2],d1[3]):[[101.0, 102.0, 102.0], [101.0, 102.0, 102.0]]",batchInput.inputIds.toString());
    }


    private static class MockBackend extends Searcher {
        String resultFile;

        MockBackend(String resultFile){
            this.resultFile = resultFile;

        }
        @Override
        public Result search(Query query, Execution execution) {
            try {
                return readTestResponse(query,resultFile);
            }catch(Exception e) {}
            return null;
        }
    }

    static Result readTestResponse(Query query,String fileName) throws FileNotFoundException, JSONException {
        Result result = new Result(query);
        String content = new Scanner(new File("src/test/resources/sample_result.json")).useDelimiter("\\Z").next();
        JSONArray passages = new JSONArray(content);
        for (int i = 0; i < passages.length(); i++) {
            JSONObject object = (JSONObject) passages.get(i);
            int id = object.getInt("id");
            String text = object.getString("text");
            JSONArray tensor = object.getJSONArray("text_token_ids");

            TensorType type = new TensorType.Builder(TensorType.Value.FLOAT).indexed("d0", tensor.length()).build();
            IndexedTensor.Builder builder = IndexedTensor.Builder.of(type);
            for (int j = 0; j < tensor.length(); j++)
                builder.cell(tensor.getInt(j), j);

            Hit hit = new Hit(Integer.toString(id), 0.2);
            hit.setField("id", id);
            hit.setField("text", text);
            hit.setField("text_token_ids", builder.build());
            result.hits().add(hit);
        }
        result.setTotalHitCount(result.hits().size());
        return result;
    }

}
