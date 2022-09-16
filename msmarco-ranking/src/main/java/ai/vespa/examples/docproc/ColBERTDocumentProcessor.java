//Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.docproc;

import ai.vespa.examples.colbert.ColbertConfig;
import ai.vespa.examples.searcher.TensorInput;
import ai.vespa.models.evaluation.Model;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.google.inject.Inject;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.FieldValue;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.language.process.Embedder;
import com.yahoo.language.wordpiece.WordPieceEmbedder;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.functions.Reduce;

import java.util.*;


public class ColBERTDocumentProcessor extends DocumentProcessor {

    private final TensorType colbertTensorType;
    private final WordPieceEmbedder tokenizer;
    private final Model colbertModel;
    private static final Tensor BATCH_TENSOR = Tensor.from("tensor<float>(d0[1]):[1]");
    private static final String modelName = "colbert_encoder";
    private int maxLength;
    private int dim;

    @Inject
    public ColBERTDocumentProcessor(ModelsEvaluator evaluator,
                                    WordPieceEmbedder tokenizer, ColbertConfig config) {
        this.tokenizer = tokenizer;
        this.maxLength = config.max_query_length();
        this.dim = config.dim();
        this.colbertModel = evaluator.requireModel(modelName);
        this.colbertTensorType = new TensorType.Builder(TensorType.Value.BFLOAT16).
                mapped("dt").indexed("x",config.dim()).build();
    }

    @Override
    public DocumentProcessor.Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document doc = put.getDocument();
                if(!doc.getDataType().getName().equals("passage")){
                    return Progress.DONE;
                }

                FieldValue e = doc.getFieldValue("dt");
                //only produce colbert document embedding if there is not already an embedding
                if (e != null)
                    continue;

                FieldValue text = doc.getFieldValue("text");
                FieldValue title = doc.getFieldValue("title");


                if (text != null && text.getDataType() != DataType.STRING)
                    return DocumentProcessor.Progress.FAILED.withReason(("Input field 'text' is not string field"));

                if (title != null && title.getDataType() != DataType.STRING)
                    return DocumentProcessor.Progress.FAILED.withReason(("Input field 'title' is not string field"));
                Tensor dt = getColBERTDocumentEmbeddings((StringFieldValue)title ,(StringFieldValue) text);
                doc.setFieldValue("dt",new TensorFieldValue(dt));
            }
        }
        return DocumentProcessor.Progress.DONE;
    }

    public Tensor getColBERTDocumentEmbeddings(StringFieldValue title, StringFieldValue text)  {
        final int D_TOKEN_ID = 2; // [unused1] token id used during training to differentiate query versus document.
        final int CLS_TOKEN_ID = 101;
        final int SEP_TOKEN_ID = 102;

        StringBuffer buffer = new StringBuffer();
        if(title != null)
            buffer.append(title.getString());
        if(text != null)
            buffer.append(text.getString());

        List<Integer> tokenIds = tokenizer.embed(buffer.toString()
                , new Embedder.Context("d")).
                stream().filter(token -> !PUNCTUATION_TOKEN_IDS.contains(token)).toList();

        if(tokenIds.size() > maxLength -3 )
            tokenIds = tokenIds.subList(0, maxLength-3);

        List<Integer> input_ids = new ArrayList<>(maxLength);
        input_ids.add(CLS_TOKEN_ID);
        input_ids.add(D_TOKEN_ID);
        input_ids.addAll(tokenIds);
        input_ids.add(SEP_TOKEN_ID);
        List<Integer> attention_mask = input_ids.stream().map(t -> t > 0 ? 1:0).toList();
        Tensor input_ids_batch = TensorInput.getTensorRepresentation(input_ids,"d1").
                multiply(BATCH_TENSOR);
        Tensor attention_mask_batch = TensorInput.getTensorRepresentation(attention_mask,"d1").
                multiply(BATCH_TENSOR);

        IndexedTensor result = (IndexedTensor)this.colbertModel.evaluatorOf().
                bind("input_ids",input_ids_batch).
                bind("attention_mask",attention_mask_batch).evaluate();
        Tensor t = result.reduce(Reduce.Aggregator.min, "d0");
        Tensor.Builder builder = Tensor.Builder.of(colbertTensorType);
        for (int token = 0; token < input_ids.size(); token++)
            for (int i = 0; i < dim; i++)
                builder.cell(TensorAddress.of(token, i), t.get(TensorAddress.of(token, i)));
        return builder.build();
    }

    private static final Set<Integer> PUNCTUATION_TOKEN_IDS = new HashSet<>(
            Arrays.asList(999, 1000, 1001, 1002, 1003, 1004, 1005, 1006,
            1007, 1008, 1009, 1010, 1011, 1012, 1013, 1024,
            1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032,
            1033, 1034, 1035, 1036, 1063, 1064, 1065, 1066));

}
