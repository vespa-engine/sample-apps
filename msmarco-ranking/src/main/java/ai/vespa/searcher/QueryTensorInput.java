//Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;
import java.util.ArrayList;
import java.util.List;

import com.yahoo.search.query.Properties;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.TensorType;

/**
 * Utility to be able to tokenize the query only once and
 * pass an instance of this class through the searcher chains
 * See <a href="https://docs.vespa.ai/en/searcher-development.html#passing-information-between-searchers">passing
 * information between searchers</a>
 */


public class QueryTensorInput {

    protected static final String NAME = "QueryTensorInput";
    private final List<Integer> queryTokenIds;

    /**
     * @param queryTokenIds The parsed bert token_ids of the query
     */
    public QueryTensorInput(List<Integer> queryTokenIds) {
        this.queryTokenIds = queryTokenIds;
    }

    /**
     * Get the Bert token_ids representation of the query
     * @return
     */
    public List<Integer> getQueryTokenIds()  {
        return queryTokenIds;
    }

    /**
     * Get the bert token_ids representation but padded with
     * @param maxLength padding up to length
     * @param padding padding number id (0 = PAD)
     * @return padded list of token ids
     */

    public List<Integer> getQueryTokenIdsPadded(int maxLength,int padding) {
        List<Integer> padded = new ArrayList<>(maxLength);
        int size = queryTokenIds.size();
        size = size < maxLength? size:maxLength;
        padded.addAll(queryTokenIds.subList(0,size));
        for(int pad = size;pad < maxLength;pad++)
            padded.add(padding);
        return padded;
    }
    /**
     * Convert List to dense representation with length input.size()
     * @param queryTokenIds the List of integers to convert to dense tensor representation
     * @return Indexed Dense Tensor with dims queryTokenIds.size()
     */

    public IndexedTensor getTensorRepresentation(List<Integer> queryTokenIds,String dimension)  {
        int size = queryTokenIds.size();
        TensorType type = new TensorType.Builder(TensorType.Value.FLOAT).
                indexed(dimension, size).build();
        IndexedTensor.Builder builder = IndexedTensor.Builder.of(type);
        for (int i = 0; i < size; ++i) {
            builder.cell(queryTokenIds.get(i), i);
        }
        return builder.build();
    }

    public static void setTo(Properties properties, QueryTensorInput value) {
        properties.set(NAME, value);
    }

    @SuppressWarnings("unchecked")
    public static QueryTensorInput getFrom(Properties properties) {
        return (QueryTensorInput) properties.get(NAME);
    }

}
