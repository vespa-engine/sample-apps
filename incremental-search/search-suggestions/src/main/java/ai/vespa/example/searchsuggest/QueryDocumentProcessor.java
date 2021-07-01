package ai.vespa.example.searchsuggest;


import com.google.inject.Inject;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.DocumentUpdate;
import com.yahoo.document.datatypes.FieldValue;
import com.yahoo.document.update.FieldUpdate;
import com.yahoo.document.update.ValueUpdate;
import com.yahoo.documentapi.*;


import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.logging.Logger;

public class QueryDocumentProcessor extends DocumentProcessor {
    private static final Logger logger = Logger.getLogger(QueryDocumentProcessor.class.getName());

    private static final String TERM_DOCUMENT_TYPE  = "term";
    private final List<String> blockWords;


    @Inject
    public QueryDocumentProcessor(BlocklistConfig config) {
        this.blockWords = config.blocklist();
    }

    public QueryDocumentProcessor() {
        //default constructor typically used for tests
        this.blockWords = new ArrayList<>();
    }

    @Override
    public Progress process(Processing processing) {
        logger.info("In process");
        for (DocumentOperation op : processing.getDocumentOperations()) {
            //if op is DocumentPut
            if (op instanceof DocumentPut) {
                //gets the document
                DocumentPut put = (DocumentPut) op;
                Document document = put.getDocument();
                if (document.getDataType().isA(TERM_DOCUMENT_TYPE)) {
                    //checking if query contains anny of the blocked words
                    if (containsBlockWords(document.getFieldValue("term"))) {
                        processing.getDocumentOperations().clear();
                        return Progress.DONE;
                    }
                }
            } else if (op instanceof DocumentUpdate) {
                DocumentUpdate update = (DocumentUpdate) op;
                if (update.getDocumentType().isA(TERM_DOCUMENT_TYPE)) {
                    FieldUpdate fieldUpdate = update.getFieldUpdate("term");
                    for (ValueUpdate<?> valueUpdate : fieldUpdate.getValueUpdates()) {
                        if (containsBlockWords(valueUpdate.getValue())) {
                            processing.getDocumentOperations().clear();
                            return Progress.DONE;
                        }
                    }
                }
            }
        }
        logger.info("  found no blocking words");
        return Progress.DONE;
    }

    private Boolean containsBlockWords(FieldValue termValue) {
        logger.info("  Searching through term");
        String query = termValue.toString().toLowerCase();
        String[] terms = query.split("\\s+");
        for (String blockWord : blockWords) {
            for (String term : terms) {
                //goes through and checks if query contains any of the blockWords
                if (Objects.equals(blockWord.toLowerCase(), term)) {
                    logger.info("  found blocking word");
                    return true;
                }
            }
        }
        return false;
    }

    @Override
    public void deconstruct() {
        super.deconstruct();
    }
}
