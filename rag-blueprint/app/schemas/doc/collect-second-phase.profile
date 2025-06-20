rank-profile collect-second-phase inherits collect-training-data {
    match-features {
            bm25(title)
            bm25(chunks)
            closeness(title_embedding)
            closeness(chunk_embeddings)
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
            # -3.5974-0.0172*bm25(chunks)+0.5504*bm25(title)-0.0005*closeness(chunk_embeddings)-0.0029*closeness(title_embedding)-0.0005*max_chunk_sim_scores+0.7143*max_chunk_text_scores
            -3.5974 -
            0.0172 * bm25(chunks) +
            0.5504 * bm25(title) -
            0.0005 * closeness(chunk_embeddings) -
            0.0029 * closeness(title_embedding) -
            0.0005 * max_chunk_sim_scores() +
            0.7143 * max_chunk_text_scores()
        }
    }

} 