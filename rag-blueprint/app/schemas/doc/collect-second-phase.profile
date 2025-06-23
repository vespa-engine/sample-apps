rank-profile collect-second-phase inherits collect-training-data {
    match-features {
            bm25(title)
            bm25(chunks)
            max_chunk_sim_scores
            max_chunk_text_scores
            avg_top_3_chunk_sim_scores
            avg_top_3_chunk_text_scores
            modified_freshness
            is_favorite
            open_count
        }
    rank-properties {
        freshness(modified_timestamp).maxAge: 94672800 # 3 years in seconds
    }
    function modified_freshness() {
        expression: freshness(modified_timestamp)
    }

    function is_favorite() {
        expression: if(attribute(favorite), 1.0, 0.0)
    }

    function open_count() {
        expression: attribute(open_count)
    }

    first-phase {
        expression {
            -7.798639+13.383840*avg_top_3_chunk_sim_scores+0.203145*avg_top_3_chunk_text_scores+0.159914*bm25(chunks)+0.191867*bm25(title)+10.067169*max_chunk_sim_scores+0.153392*max_chunk_text_scores
        }
    }

} 