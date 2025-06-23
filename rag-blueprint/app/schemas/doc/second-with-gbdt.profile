rank-profile second-with-gbdt inherits collect-second-phase {
    match-features {
        max_chunk_sim_scores
        max_chunk_text_scores
        avg_top_3_chunk_text_scores
        avg_top_3_chunk_sim_scores
        bm25(title)
        modified_freshness
        open_count
        firstPhase
    }
    # nativeProximity,146.9688407897949
    # max_chunk_sim_scores,129.52473888397216
    # avg_top_3_chunk_sim_scores,111.75787982940673
    # nativeFieldMatch,15.843959999084472
    # firstPhase,5.440619850158692
    # avg_top_3_chunk_text_scores,3.9279340744018554
    # elementSimilarity(chunks),0.8898500204086304
    # fieldMatch(title).earliness,0.48935999870300295
    # fieldMatch(title).longestSequenceRatio,0.23010001182556153
    # modified_freshness,0.19112980365753174
    # bm25(title),0.0531499981880188
    # open_count,0.026539599895477294
    rank-features {
        nativeProximity
        nativeFieldMatch
        nativeRank
        elementSimilarity(chunks)
        fieldMatch(title).earliness
        fieldMatch(title).longestSequenceRatio 
    }
    second-phase {
        expression: lightgbm("lightgbm_model.json")
    }

    summary-features: top_3_chunk_sim_scores
}