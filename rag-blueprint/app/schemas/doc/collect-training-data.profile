rank-profile collect-training-data {
        match-features {
            bm25(title)
            bm25(chunks)
            closeness(title_embedding)
            closeness(chunk_embeddings)
            max_chunk_sim_scores
            max_chunk_text_scores
            avg_top_3_chunk_sim_scores
            avg_top_3_chunk_text_scores
        }
        inputs {
            query(embedding) tensor<int8>(x[96])
        }

        rank chunks {
            element-gap: 0 # Fixed length chunking should not cause any positional gap between elements
        }
        function chunk_text_scores() {
            expression: elementwise(bm25(chunks),chunk,float)
        }

        function chunk_dist_scores() {
            expression: reduce(hamming(query(embedding), attribute(chunk_embeddings)), sum, x)
        }

        function chunk_sim_scores() {
            expression: 1/ (1 + chunk_dist_scores())
        }

        function top_3_chunk_text_scores() {
            expression: top(3, chunk_text_scores())
        }

        function top_3_chunk_sim_scores() {
            expression: top(3, chunk_sim_scores())
        }


        function avg_top_3_chunk_text_scores() {
            expression: reduce(top_3_chunk_text_scores(), avg, chunk)
        }
        function avg_top_3_chunk_sim_scores() {
            expression: reduce(top_3_chunk_sim_scores(), avg, chunk)
        }
        
        function max_chunk_text_scores() {
            expression: reduce(chunk_text_scores(), max, chunk)
        }

        function max_chunk_sim_scores() {
            expression: reduce(chunk_sim_scores(), max, chunk)
        }

        first-phase {
            expression {
                closeness(title_embedding) +
                closeness(chunk_embeddings) +
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