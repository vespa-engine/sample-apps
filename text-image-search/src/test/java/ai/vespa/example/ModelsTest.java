// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

package ai.vespa.example;

import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.config.FileReference;
import com.yahoo.searchlib.rankingexpression.parser.ParseException;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ModelsTest {

    private static BPETokenizer tokenizer;

    @BeforeAll
    static void setup() {
        FileReference file = new FileReference("src/main/application/files/bpe_simple_vocab_16e6.txt.gz");
        BpeTokenizerConfig config = new BpeTokenizerConfig.Builder().contextlength(77).vocabulary(file).build();
        tokenizer = new BPETokenizer(config);
    }

    @Test
    public void testTokenizer() {
        assertEncodeDecode("a diagram",
                           "tensor(d0[4]):[49406, 320, 22697, 49407]");
        assertEncodeDecode("Two people are sitting next to a wood - stacked campfire at night",
                           "tensor(d0[15]):[49406, 1237, 1047, 631, 4919, 1131, 531, 320, 1704, 268, 24990, 32530, 536, 930, 49407]");
        assertEncodeDecode("A surfer is riding a wave",
                           "tensor(d0[8]):[49406, 320, 24925, 533, 6765, 320, 4535, 49407]");
        assertEncodeDecode("Two men walk through a city with words on the street",
                           "tensor(d0[13]):[49406, 1237, 1656, 2374, 1417, 320, 1305, 593, 2709, 525, 518, 2012, 49407]");
        assertEncodeDecode("An indian chief in full dress .",
                           "tensor(d0[9]):[49406, 550, 3606, 3455, 530, 1476, 2595, 269, 49407]");
        assertEncodeDecode("When did man last land on the moon ?",
                           "tensor(d0[11]):[49406, 827, 1335, 786, 952, 973, 525, 518, 3293, 286, 49407]");
    }

    @Test
    public void testModels() throws ParseException {
        ModelsEvaluator modelsEvaluator = ModelsEvaluatorTester.create("src/main/application/models");

        // The "models" directory contains 2 ONNX models: "transformer" and "visual"
        assertEquals(1, modelsEvaluator.models().size());
        assertTrue(modelsEvaluator.models().containsKey("transformer"));

        testTransformerModel(modelsEvaluator);
    }

    private void testTransformerModel(ModelsEvaluator modelsEvaluator) throws ParseException {
        FunctionEvaluator transformer = modelsEvaluator.evaluatorOf("transformer", "output");

        // The transformer model has one input, named "input" and has type "tensor(d0[],d1[])"
        assertEquals(1, transformer.function().arguments().size());
        assertEquals("input", transformer.function().arguments().get(0));
        assertEquals("tensor(d0[],d1[77])", transformer.function().argumentTypes().get("input").toString());

        Tensor input = tokenizer.encode("a diagram", 77, "d1").expand("d0");
        Tensor output = transformer.bind("input", input).evaluate();
        assertEquals("tensor<float>(d0[1],d1[512])", output.type().toString());

        // Check the value of the first cell
        assertEquals(5.4698e-02, output.get(TensorAddress.of(0,0)), 1e-5);
    }


    private void assertEncodeDecode(String text, String tensor, int contextLength) {
        Tensor expected = Tensor.from(tensor);
        Tensor tokens = tokenizer.encode(text, contextLength, "d0");
        String decoded = tokenizer.decode(tokens);
        assertEquals(expected, tokens);
        assertEquals(text.strip().toLowerCase(), decoded.replace("<|startoftext|>", "").replace("<|endoftext|>", "").strip().toLowerCase());
    }

    private void assertEncodeDecode(String text, String tensor) {
        assertEncodeDecode(text, tensor, text.split(" ").length + 2);
    }

}
