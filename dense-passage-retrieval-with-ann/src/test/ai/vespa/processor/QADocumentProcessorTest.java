package ai.vespa.processor;


import ai.vespa.tokenizer.BertModelConfig;
import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.container.StatisticsConfig;
import com.yahoo.docproc.CallStack;
import com.yahoo.docproc.Processing;
import com.yahoo.docproc.DocprocService;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.jdisc.metric.NullMetric;
import com.yahoo.document.*;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.statistics.StatisticsImpl;
import com.yahoo.tensor.TensorType;

import org.junit.Test;

import static org.junit.Assert.*;

public class QADocumentProcessorTest {

    static BertModelConfig bertModelConfig;

    static {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(new com.yahoo.config.FileReference("src/test/resources/bert-base-uncased-vocab.txt")).max_input(128);
        bertModelConfig = builder.build();
    }


    private static DocprocService setupDocprocService(DocumentProcessor processor) {
        CallStack stack = new CallStack("default", new StatisticsImpl(new StatisticsConfig(new StatisticsConfig.Builder())), new NullMetric());
        stack.addLast(processor);
        DocprocService service = new DocprocService("default");
        service.setCallStack(stack);
        service.setInService(true);
        return service;
    }

    private static Processing getProcessing(DocumentOperation... operations) {
        Processing processing = new Processing();
        for (DocumentOperation op : operations) {
            processing.addDocumentOperation(op);
        }
        return processing;
    }

    private static DocumentType createWikiDocType() {
        DocumentType type = new DocumentType("wiki");
        type.addField("title", DataType.STRING);
        type.addField("title_token_ids", TensorDataType.getTensor(TensorType.fromSpec("tensor<float>(d0[256])")));
        type.addField("text", DataType.STRING);
        type.addField("text_token_ids", TensorDataType.getTensor(TensorType.fromSpec("tensor<float>(d0[256])")));
        return type;
    }

    @Test
    public void testProcessing() throws Exception {
        Document doc = new Document(createWikiDocType(), "id:foo:wiki::0");
        doc.setFieldValue("title", new StringFieldValue("Britney_spears"));
        doc.setFieldValue("text", new StringFieldValue("Britney Jean Spears (born December 2, 1981) is an American singer, songwriter, dancer, and actress."));
        Processing p = getProcessing(new DocumentPut(doc));
        DocprocService service = setupDocprocService(new QADocumentProcessor(new BertTokenizer(bertModelConfig, new SimpleLinguistics())));
        service.getExecutor().process(p);

        TensorFieldValue title_tensor = (TensorFieldValue)doc.getFieldValue("title_token_ids");
        TensorFieldValue text_tensor = (TensorFieldValue)doc.getFieldValue("text_token_ids");
        assertNotNull(title_tensor);
        assertNotNull(text_tensor);
        assertTrue(title_tensor.toString().startsWith("tensor(d0[256]):[29168.0, 1035.0, 13957.0"));
        assertTrue(text_tensor.toString().startsWith("tensor(d0[256]):[29168.0, 3744.0, 13957.0, 1006.0, 2141.0, 2285.0, 1016.0, 1010.0"));
    }
}
