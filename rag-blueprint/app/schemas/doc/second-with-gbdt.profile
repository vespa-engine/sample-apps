rank-profile second-with-gbdt inherits collect-second-phase {
    match-features {
        max_chunk_sim_scores
        max_chunk_text_scores
        avg_top_3_chunk_text_scores
        avg_top_3_chunk_sim_scores
        closeness(chunk_embeddings)
        bm25(chunks)
        bm25(chunks)
        modified_freshness
        firstPhase
    }
    rank-features {
        nativeProximity
        nativeFieldMatch
        nativeRank
        elementCompleteness(chunks).completeness
        elementCompleteness(chunks).queryCompleteness
        elementSimilarity(chunks)        
    }
    second-phase {
        expression: lightgbm("lightgbm_model.json")
    }

    summary-features: top_3_chunk_sim_scores
}