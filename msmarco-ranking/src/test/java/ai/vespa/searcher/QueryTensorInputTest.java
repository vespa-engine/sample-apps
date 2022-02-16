// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

import com.yahoo.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class QueryTensorInputTest {

    @Test
    public void test_with_padding() {
        List<Integer> input = new ArrayList<>(Arrays.asList(2345,1234));
        QueryTensorInput tensorInput = new QueryTensorInput(input);
        List<Integer> padded = tensorInput.getQueryTokenIdsPadded(24,0);
        assertEquals(24,padded.size());
        assertEquals(2345,padded.get(0));
        assertEquals(1234,padded.get(1));
        assertEquals(0,padded.get(2));
    }

    @Test
    public void test_trim_to_max_length() {
        List<Integer> input = new ArrayList<>(Arrays.asList(2345,1234,1234));
        QueryTensorInput tensorInput = new QueryTensorInput(input);
        List<Integer> padded = tensorInput.getQueryTokenIdsPadded(1,0);
        assertEquals(1,padded.size());
        assertEquals(2345,padded.get(0));
    }

    @Test
    public void test_just_right_length() {
        List<Integer> input = new ArrayList<>(Arrays.asList(2345,1234,12345));
        QueryTensorInput tensorInput = new QueryTensorInput(input);
        List<Integer> padded = tensorInput.getQueryTokenIdsPadded(3,0);
        assertEquals(3,padded.size());
        assertEquals(2345,padded.get(0));
        assertEquals(1234,padded.get(1));
        assertEquals(12345,padded.get(2));
    }

    @Test
    public void test_input_zero_length() {
        List<Integer> input = new ArrayList<>();
        QueryTensorInput tensorInput = new QueryTensorInput(input);
        List<Integer> padded = tensorInput.getQueryTokenIdsPadded(3,0);
        assertEquals(3,padded.size());
        assertEquals(0,padded.get(0));
        assertEquals(0,padded.get(1));
        assertEquals(0,padded.get(2));
    }

    @Test
    public void test_list_to_tensor_representation() {
        List<Integer> input = new ArrayList<>(Arrays.asList(2345,1234,12345));
        QueryTensorInput tensorInput = new QueryTensorInput(input);
        Tensor tensor = tensorInput.getTensorRepresentation(input,"d0");
        assertEquals("tensor<float>(d0[3]):[2345.0, 1234.0, 12345.0]",tensor.toString());
        tensor = tensorInput.getTensorRepresentation(input,"d1");
        assertEquals("tensor<float>(d1[3]):[2345.0, 1234.0, 12345.0]",tensor.toString());
    }
}
