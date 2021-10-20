// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.cloud.docsearch;

import com.yahoo.prelude.query.PrefixItem;
import com.yahoo.prelude.query.WeakAndItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.result.Relevance;
import com.yahoo.search.searchchain.Execution;


import java.util.List;


public class DocumentationSearcher extends Searcher {


    @Override
    public Result search(Query query, Execution execution) {
        Result suggestionResult;
        Result documentResult;
        Object termProperty = query.properties().get("term");
        if (termProperty != null){
            String searchTerms = termProperty.toString();
            suggestionResult = getSuggestions(searchTerms, execution);
            execution.fill(suggestionResult);
            Query docQuery = new Query();
            docQuery.getModel().setRestrict("doc");
            WeakAndItem weakAndItem = new WeakAndItem();
            if (suggestionResult.getHitCount() > 0) {
                HitGroup hits = suggestionResult.hits();
                List<Hit> hitList = hits.asList();
                String[] topHitStringArray = getTopHitStringArray(hitList);
                docQuery.setHits(20);
                for (String term: topHitStringArray){
                    WordItem wordItem = new WordItem(term, true);
                    weakAndItem.addItem(wordItem);
                }
            } else {
                docQuery.setHits(10);
                for (String term: searchTerms.split(" ")){
                    WordItem wordItem = new WordItem(term, true);
                    weakAndItem.addItem(wordItem);
                }
            }
            docQuery.getModel().getQueryTree().setRoot(weakAndItem);
            docQuery.getRanking().setProfile("documentation");
            documentResult = execution.search(docQuery);
            execution.fill(documentResult);
            return combineHits(documentResult,suggestionResult);
        }
        return execution.search(query);
    }

    private Result getSuggestions(String searchTerms, Execution execution) {
        Query query = new Query();
        query.getModel().setRestrict("term");
        query.getModel().getQueryTree().setRoot(new PrefixItem(searchTerms, "default"));
        query.getRanking().setProfile("term_rank");
        query.setHits(10);
        Result suggestionResult = execution.search(query);
        execution.fill(suggestionResult);
        return suggestionResult;
    }

    private String[] getTopHitStringArray(List<Hit> hitList) {
        Hit topHit = hitList.get(0);
        double highestScore = 0;
        for (Hit hit: hitList){
            Relevance relevance = hit.getRelevance();
            if (highestScore < relevance.getScore()){
                topHit = hit;
                highestScore = relevance.getScore();
            }
        }
        if (topHit.fields().get("term") != null){
            return topHit.getField("term").toString().split(" ");
        }else{
            throw new RuntimeException("There were no field called \"term\"");
        }
    }

    private Result combineHits(Result result1, Result result2){
        HitGroup hits1 = result1.hits();
        HitGroup hits2 = result2.hits();
        for (Hit hit: hits2){
            hits1.add(hit);
        }
        return result1;
    }




}
