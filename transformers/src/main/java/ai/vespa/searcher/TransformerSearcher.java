// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;

import com.google.inject.Inject;
import com.yahoo.language.Linguistics;
import com.yahoo.language.process.Embedder;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.language.wordpiece.WordPieceEmbedder;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.WeakAndItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import java.util.List;


public class TransformerSearcher extends Searcher {

    private static final String QUERY_TENSOR_NAME = "query(input)";
    private static final TensorType INPUT_TENSOR_TYPE = TensorType.fromSpec("tensor<float>(d0[32])");

    private WordPieceEmbedder embedder;
    private Linguistics linguistics;

    @Inject
    public TransformerSearcher(WordPieceEmbedder embedder, Linguistics linguistics) {
        this.embedder = embedder;
        this.linguistics = linguistics;
    }

    @Override
    public Result search(Query query, Execution execution) {
        String queryString = query.getModel().getQueryString();
        if (queryString == null || queryString.length() == 0)
            return new Result(query, ErrorMessage.createBadRequest("No query input"));

        var bm25Query = createBM25Query(queryString, query);
        var tokenSequence = createTokenSequence(queryString);

        query.getModel().getQueryTree().setRoot(bm25Query);
        query.getRanking().getFeatures().put(QUERY_TENSOR_NAME, tokenSequence);
        query.getRanking().setProfile("transformer");

        return execution.search(query);
    }

    private Item createBM25Query(String queryString, Query query) {
        WeakAndItem wand = new WeakAndItem();
        wand.setN(query.getHits());
        Iterable<Token> tokens = linguistics.getTokenizer().
                tokenize(queryString, query.getModel().getLanguage(), StemMode.NONE, true);
        for (Token t : tokens) {
            if (t.isIndexable())
                wand.addItem(new WordItem(t.getTokenString(), "default", true));
        }
        return wand;
    }

    private Tensor createTokenSequence(String queryString) {
        int maxLength = INPUT_TENSOR_TYPE.sizeOfDimension("d0").get().intValue();
        Tensor.Builder builder = Tensor.Builder.of(INPUT_TENSOR_TYPE);
        List<Integer> tokens = embedder.embed(queryString,new Embedder.Context("q"));
        int i = 0;
        for (Integer tokenId :tokens) {
            if (i == maxLength)
                break;
            builder.cell(tokenId, i++);
        }
        return builder.build();
    }

}

