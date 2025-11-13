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
    
    # When decaying scores by how old documents are (see below),
    # get to 0.0 after 3 years, then remain at 0.0.
    rank-properties {
        freshness(modified_timestamp).maxAge: 94672800 # 3 years in seconds
    }
    # If modified_timestamp is now, freshness is 1.0.
    # Otherwise, decay linearly towards 0.0 at a rate of 1/3 per year (as configured above).
    function modified_freshness() {
        expression: freshness(modified_timestamp)
    }

    # Returns 1.0 if the document is a favorite, 0.0 otherwise.
    function is_favorite() {
        expression: if(attribute(favorite), 1.0, 0.0)
    }

    # Returns the value of the open_count field.
    function open_count() {
        expression: attribute(open_count)
    }

    first-phase {
        expression {
            -7.798639+13.383840*avg_top_3_chunk_sim_scores+0.203145*avg_top_3_chunk_text_scores+0.159914*bm25(chunks)+0.191867*bm25(title)+10.067169*max_chunk_sim_scores+0.153392*max_chunk_text_scores
        }
    }

} 