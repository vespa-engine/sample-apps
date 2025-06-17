rank-profile learned-linear inherits collect-training-data {
        match-features: 
        inputs {
            query(embedding) tensor<int8>(x[96])
            query(intercept) double
            query(bm25_chunks_param) double
            query(bm25_title_param) double
            query(closeness_chunk_embeddings_param) double
            query(closeness_title_embedding_param) double
            query(max_chunk_sim_scores_param) double
            query(max_chunk_text_scores_param) double
        }
        first-phase {
            expression {
                query(intercept) + 
                query(closeness_title_embedding_param) * closeness(title_embedding) + 
                query(closeness_chunk_embeddings_param) * closeness(chunk_embeddings) +
                query(bm25_title_param) * bm25(title) + 
                query(bm25_chunks_param) * bm25(chunks) +
                query(max_chunk_sim_scores_param) * max_chunk_sim_scores() +
                query(max_chunk_text_scores_param) * max_chunk_text_scores()
            }
        }
        summary-features {
            top_3_chunk_sim_scores
        }
        
    }