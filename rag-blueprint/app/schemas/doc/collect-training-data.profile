rank-profile collect-training-data inherits base-features {
        match-features {
            bm25(title)
            bm25(chunks)
            max_chunk_sim_scores
            max_chunk_text_scores
            avg_top_3_chunk_sim_scores
            avg_top_3_chunk_text_scores

        }


        first-phase {
            expression {
                # Not used in this profile
                bm25(title) + 
                bm25(chunks) +
                max_chunk_sim_scores() +
                max_chunk_text_scores()
            }
        }

        second-phase {
            expression: random
        }
    }