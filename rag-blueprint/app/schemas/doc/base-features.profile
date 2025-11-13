rank-profile base-features {
        inputs {
            query(embedding) tensor<int8>(x[96])
            query(float_embedding) tensor<float>(x[768])
        }

        # Fixed length chunking should not cause any positional gap between elements
        # => lexical search (e.g., chunks contains 'hello world') works as if we had a big blob of text,
        # (e.g., would match ["and she said hello", "world!"])
        rank chunks {
            element-gap: 0
        }

        # Creates a tensor with a single mapped dimension (i.e., the chunk ID)
        # and the BM25 score for each chunk. Returns something like:
        #{
        #    "type": "tensor(chunk{})",
        #    "cells": {
        #        "2": 0.5112776,
        #        "5": 1.1021805
        #    }
        #}
        function chunk_text_scores() {
            expression: elementwise(bm25(chunks),chunk,float)
        }

        # Unpacks the 8 bits of each integer from every chunk into 8 floats representing their values
        # effectively transforms tensor<int8>(chunk{}, x[96]) into tensor<float>(chunk{}, x[768]).
        # This prepares the bit embeddings for dot product with the query embedding. Returns something like:
        #{
        #    "type": "tensor(chunk{})",
        #    "cells": {
        #        "2": [0.0, 1.0, 0.0, 0.0, 1.0, ... up to 768 floats],
        #        "5": [1.0, 0.0, 1.0, 0.0, 1.0, ... up to 768 floats]
        #    }
        #}
        function chunk_emb_vecs() {
            expression: unpack_bits(attribute(chunk_embeddings))
        }

        # Computes the dot product of the query embedding with each chunk embedding.
        # Returns a tensor with a similarity score for each chunk ID. E.g.,
        #{
        #    "type": "tensor(chunk{})",
        #    "cells": {
        #        "2": 8.7,   # e.g., this is the dot product of the query embedding with the chunk embedding for chunk ID 2
        #        "5": 1.2
        #    }
        #}
        function chunk_dot_prod() {
            expression: reduce(query(float_embedding) * chunk_emb_vecs(), sum, x)
        }


        # Computes the magnitude (Euclidean norm) of all vectors in a tensor.
        # We will use this to normalize (i.e., bring back to 0-1 range) the chunk dot product,
        # which will tend to be higher for embeddings with more dimensions. This normalized
        # dot product is the cosine similarity.
        function vector_norms(t) {
            expression: sqrt(sum(pow(t, 2), x))
        }

        # Computes the cosine similarity between the query embedding and each chunk embedding,
        # by dividing the dot product by the product of the magnitudes of the two vectors.
        #
        # Returns a tensor with a similarity score for each chunk ID. E.g.,
        #{
        #    "type": "tensor(chunk{})",
        #    "cells": {
        #        "2": 0.98,   # e.g., this is the cosine similarity between the query embedding and the chunk embedding for chunk ID 2
        #        "5": 0.75    # notice how values are normalized to 0-1 range, unlike the dot product
        #    }
        #}
        function chunk_sim_scores() {
            expression: chunk_dot_prod() / (vector_norms(chunk_emb_vecs()) * vector_norms(query(float_embedding)))
        }

        # Returns a tensor with the top 3 chunk IDs by their BM25 lexical scores. E.g.,
        #{
        #    "type": "tensor(chunk{})",
        #    "cells": {
        #        "3": 3.8021805,
        #        "5": 1.1021805,
        #        "2": 0.5112776
        #    }
        #}
        function top_3_chunk_text_scores() {
            expression: top(3, chunk_text_scores())
        }

        # Returns a tensor with the top 3 chunk IDs by their cosine similarity scores.
        function top_3_chunk_sim_scores() {
            expression: top(3, chunk_sim_scores())
        }

        # Returns the average of the top 3 chunks' BM25 scores.
        function avg_top_3_chunk_text_scores() {
            expression: reduce(top_3_chunk_text_scores(), avg, chunk)
        }

        # Returns the average of the top 3 chunks' cosine similarity scores.
        function avg_top_3_chunk_sim_scores() {
            expression: reduce(top_3_chunk_sim_scores(), avg, chunk)
        }
        
        # Returns the maximum of the chunk BM25 lexical scores.
        function max_chunk_text_scores() {
            expression: reduce(chunk_text_scores(), max, chunk)
        }

        # Returns the maximum of the chunk cosine similarity scores.
        function max_chunk_sim_scores() {
            expression: reduce(chunk_sim_scores(), max, chunk)
        }
}