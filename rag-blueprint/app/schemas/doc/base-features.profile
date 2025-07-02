rank-profile base-features {
        inputs {
            query(embedding) tensor<int8>(x[96])
            query(float_embedding) tensor<float>(x[768])
        }

        rank chunks {
            element-gap: 0 # Fixed length chunking should not cause any positional gap between elements
        }
        function chunk_text_scores() {
            expression: elementwise(bm25(chunks),chunk,float)
        }

        function chunk_emb_vecs() {
            expression: unpack_bits(attribute(chunk_embeddings))
        }

        function chunk_dot_prod() {
            expression: reduce(query(float_embedding) * chunk_emb_vecs(), sum, x)
        }

        function vector_norms(t) {
            expression: sqrt(sum(pow(t, 2), x))
        }
        function chunk_sim_scores() {
            expression: chunk_dot_prod() / (vector_norms(chunk_emb_vecs()) * vector_norms(query(float_embedding)))
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
}