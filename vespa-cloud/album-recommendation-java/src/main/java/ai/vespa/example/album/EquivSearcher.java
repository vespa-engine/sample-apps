// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.album;

import com.yahoo.prelude.query.CompositeItem;
import com.yahoo.prelude.query.EquivItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.PhraseItem;
import com.yahoo.prelude.query.TermItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.query.QueryTree;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.yolean.chain.After;

import java.util.List;
import java.util.Map;

@After("MinimalQueryInserter")
public class EquivSearcher extends Searcher {

    Map<String, List<String>> artistSpellings = Map.of(
            "metallica", List.of("metalica", "metallika"),
            "rammstein", List.of("ramstein", "raamstein"));

    public EquivSearcher() { }

    @Override
    public Result search(Query query, Execution execution) {
        QueryTree tree = query.getModel().getQueryTree();
        Item rootItem = tree.getRoot();
        rootItem = equivize(rootItem);
        tree.setRoot(rootItem);

        return execution.search(query);
    }

    private Item equivize(Item item) {
        if (item instanceof TermItem) {
            String word = ((TermItem) item).stringValue();
            String indexName = ((TermItem) item).getIndexName();
            if(indexName.equals("artist")) {
                List<String> synonyms = artistSpellings.get(word);
                if (synonyms != null) {
                    EquivItem eq = new EquivItem(item, synonyms);
                    return eq;
                }
            }
        } else if (item instanceof PhraseItem) {
            // cannot put EQUIV inside PHRASE
            return item;
        } else if (item instanceof CompositeItem) {
            CompositeItem cmp = (CompositeItem)item;
            for (int i = 0; i < cmp.getItemCount(); ++i) {
                cmp.setItem(i, equivize(cmp.getItem(i)));
            }
            return cmp;
        }
        return item;
    }
}
