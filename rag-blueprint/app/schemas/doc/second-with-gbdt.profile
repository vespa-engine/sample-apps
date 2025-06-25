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
    # nativeProximity,168.84977385997772
    # firstPhase,151.73823466300965
    # max_chunk_sim_scores,69.43774781227111
    # avg_top_3_chunk_text_scores,56.507930064201354
    # avg_top_3_chunk_sim_scores,31.87002867460251
    # nativeRank,20.071615393646063
    # nativeFieldMatch,15.991393876075744
    # elementSimilarity(chunks),9.700291919708253
    # bm25(chunks),3.8777143508195877
    # max_chunk_text_scores,3.6405647873878477
    # "fieldTermMatch(chunks,4).firstPosition",1.2615019798278808
    # "fieldTermMatch(chunks,4).occurrences",1.0542740106582642
    # "fieldTermMatch(chunks,4).weight",0.7263560056686401
    # term(3).significance,0.5077840089797974
    rank-features {
        nativeProximity
        nativeFieldMatch
        nativeRank
        elementSimilarity(chunks)
        fieldTermMatch(chunks, 4).firstPosition
        fieldTermMatch(chunks, 4).occurrences
        fieldTermMatch(chunks, 4).weight
        term(3).significance
    }
    second-phase {
        expression: lightgbm("lightgbm_model.json")
    }

    summary-features: top_3_chunk_sim_scores
}