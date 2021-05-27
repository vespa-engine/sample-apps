// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;

import ai.vespa.tokenizer.BertModelConfig;
import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.component.chain.Chain;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.Query;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import java.io.IOException;


public class QASearcherTest {

    static BertModelConfig bertModelConfig;
    static BertTokenizer tokenizer;

    static {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(new com.yahoo.config.FileReference("src/test/resources/bert-base-uncased-vocab.txt")).max_input(128);
        bertModelConfig = builder.build();
        try {
            tokenizer = new BertTokenizer(bertModelConfig, new SimpleLinguistics());
        } catch (IOException e) {
        }
    }


    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }


    private static class MockBackend extends Searcher {
        @Override
        public Result search(Query query, Execution execution) {
            if(isEncoderQuery(query)) {
                Result result = execution.search(query);
                result.setTotalHitCount(1);
                Tensor.Builder b = Tensor.Builder.of(TensorType.fromSpec("tensor<float>(x[768])"));
                for (long i = 0; i < 768; i++)
                    b.cell(Math.random(), i);
            }
            return execution.search(query);
        }
        private boolean isEncoderQuery(Query query) {
            return query.getModel().getRestrict().equals("query");
        }
    }

}


