// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.searcher;

import ai.vespa.examples.BPETokenizer;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.component.chain.Chain;
import com.yahoo.config.FileReference;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.apache.hc.core5.net.URIBuilder;
import org.junit.jupiter.api.Test;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestClipEmbeddingSearcher {

    @Test
    public void test() {
        FileReference file = new FileReference("src/main/application/files/bpe_simple_vocab_16e6.txt.gz");
        ai.vespa.examples.BpeTokenizerConfig config =
                new ai.vespa.examples.BpeTokenizerConfig.Builder().contextlength(77).vocabulary(file).build();
        BPETokenizer tokenizer = new BPETokenizer(config);
        ModelsEvaluator modelsEvaluator = ModelsEvaluatorTester.create("src/main/application/models");

        CLIPEmbeddingSearcher searcher = new CLIPEmbeddingSearcher(modelsEvaluator, tokenizer);

        URIBuilder builder = searchUri();
        builder.setParameter(
                "yql",
                "select * from sources * where foo=1");
        builder.setParameter(
                "prompt",
                "kneeling cat knight, portrait," +
                        " finely detailed armor, intricate design, silver, silk, cinematic lighting, 4k");

        Query query = new Query(builder.toString());

        execute(query, searcher);
        Optional<Tensor> t = query.getRanking().getFeatures().getTensor("query(q)");
        assertTrue(t.isPresent());
        Tensor embedding = t.get();
        assertEquals("tensor<float>(x[768])", embedding.type().toString());
    }

    private static Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }

    private static URIBuilder searchUri() {
        URIBuilder builder = new URIBuilder();
        builder.setPath("search/");
        return builder;
    }
}
