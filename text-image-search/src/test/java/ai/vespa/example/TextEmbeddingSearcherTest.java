// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

package ai.vespa.example;

import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.component.chain.Chain;
import com.yahoo.config.FileReference;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TextEmbeddingSearcherTest {

    @Test
    public void test() {
        FileReference file = new FileReference("src/main/application/files/bpe_simple_vocab_16e6.txt.gz");
        BpeTokenizerConfig config = new BpeTokenizerConfig.Builder().contextlength(77).vocabulary(file).build();
        BPETokenizer tokenizer = new BPETokenizer(config);
        ModelsEvaluator modelsEvaluator = ModelsEvaluatorTester.create("src/main/application/models");

        TextEmbeddingSearcher searcher = new TextEmbeddingSearcher(modelsEvaluator, tokenizer);

        Query query = new Query("/search/?input=a+diagram");

        Result r = execute(query, searcher);
        Optional<Tensor> t = query.getRanking().getFeatures().getTensor("query(vit_b_32_text)");
        assertTrue(t.isPresent());

        Tensor embedding = t.get();
        assertEquals("tensor<float>(x[512])", embedding.type().toString());

        assertEquals(4.6018e-03, embedding.get(TensorAddress.of(0)), 1e-5);
    }

    private static Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }

}
