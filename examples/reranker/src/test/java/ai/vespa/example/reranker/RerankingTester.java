// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.reranker;

import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.component.chain.Chain;
import com.yahoo.container.jdisc.HttpRequest;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;

import java.util.Map;

/**
 * Reranking unit test utility.
 *
 * @author bratseth
 */
public class RerankingTester {

    final MockVespaClient client = new MockVespaClient();
    final RerankerConfig rerankerConfig = new RerankerConfig.Builder().endpoint("my-endpoint")
                                             .rerank(new RerankerConfig.Rerank.Builder().hits(10)
                                                                                        .profile("my-profile")
                                                                                        .model("xgboost_model_example"))
                                             .build();
    final ModelsEvaluator evaluator = ModelsEvaluatorTester.create("src/main/application/models");
    final RerankingSearcher rerankingSearcher = new RerankingSearcher(rerankerConfig, evaluator);
    final VespaSearcher vespaSearcher = new VespaSearcher(client);

    public Execution executionOf(Searcher... searcher) {
        return new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
    }

    /** Returns an exedcutor of the full reranker chain. */
    public Execution execution() {
        return new Execution(new Chain<>(rerankingSearcher, vespaSearcher), Execution.Context.createContextStub());
    }

    static class MockVespaClient extends VespaClient {

        HttpRequest lastRequest;
        Map<String, Object> lastOverridingProperties;

        MockVespaClient() {
            super("ignored");
        }

        @Override
        public Response search(HttpRequest request, Map<String, Object> overridingProperties) {
            this.lastRequest = request;
            this.lastOverridingProperties = overridingProperties;
            return new Response(200, exampleResultJson);
        }

    }

    static final String exampleResultJson = """
    {
    "root": {
        "id": "toplevel",
        "relevance": 1.0,
        "fields": {
            "totalCount": 3
        },
        "coverage": {
            "coverage": 100,
            "documents": 3,
            "full": true,
            "nodes": 1,
            "results": 1,
            "resultsFull": 1
        },
        "children": [
            {
                "id": "id:mynamespace:music::love-id-here-to-stay",
                "relevance": 0.3,
                "source": "music",
                "fields": {
                    "sddocname": "music",
                    "documentid": "id:mynamespace:music::love-id-here-to-stay",
                    "artist": "Diana Krall",
                    "album": "Love Is Here To Stay",
                    "year": 2018,
                    "summaryfeatures": {
                        "fieldMatch(artist).completeness": 1.0,
                        "fieldMatch(artist).proximity": 0.9,
                        "vespa.summaryFeatures.cached": 0.0
                    }

                }
            },
            {
                "id": "id:mynamespace:music::hardwired-to-self-destruct",
                "relevance": 0.2,
                "source": "music",
                "fields": {
                    "sddocname": "music",
                    "documentid": "id:mynamespace:music::hardwired-to-self-destruct",
                    "artist": "Metallica",
                    "album": "Hardwired...To Self-Destruct",
                    "year": 2016,
                    "summaryfeatures": {
                        "fieldMatch(artist).completeness": 0.4,
                        "fieldMatch(artist).proximity": 0.3,
                        "vespa.summaryFeatures.cached": 0.0
                    }
                }
            },
            {
                "id": "id:mynamespace:music::a-head-full-of-dreams",
                "relevance": 0.1,
                "source": "music",
                "fields": {
                    "sddocname": "music",
                    "documentid": "id:mynamespace:music::a-head-full-of-dreams",
                    "artist": "Coldplay",
                    "album": "A Head Full of Dreams",
                    "year": 2015,
                    "summaryfeatures": {
                        "fieldMatch(artist).completeness": 0.25,
                        "fieldMatch(artist).proximity": 0.15,
                        "vespa.summaryFeatures.cached": 0.05
                    }
                }
            }
        ]
    }
    }""";



}
