package ai.vespa.cloud.docsearch;

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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.logging.Logger;

public class QueryDocumentProcessor extends DocumentProcessor {
    private static final Logger logger = Logger.getLogger(QueryDocumentProcessor.class.getName());

    private static final String TERM_DOCUMENT_TYPE  = "term";
    private static final String filePath = "files/accepted_words.txt";
    private final HashSet<String> acceptedWords;


    @Inject
    public QueryDocumentProcessor() {
        this.acceptedWords = getAcceptedWords();
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
                    //checking if query contains only accepted words
                    if (!containsOnlyAcceptedWords(document.getFieldValue("term"))) {
                        processing.getDocumentOperations().clear();
                        return Progress.DONE;
                    }
                }
            } else if (op instanceof DocumentUpdate) {
                DocumentUpdate update = (DocumentUpdate) op;
                if (update.getDocumentType().isA(TERM_DOCUMENT_TYPE)) {
                    FieldUpdate fieldUpdate = update.getFieldUpdate("term");
                    if (fieldUpdate != null) {
                        for (ValueUpdate<?> valueUpdate : fieldUpdate.getValueUpdates()) {
                            if (!containsOnlyAcceptedWords(valueUpdate.getValue())) {
                                processing.getDocumentOperations().clear();
                                return Progress.DONE;
                            }
                        }
                    }
                }
            }
        }
        logger.info("  found no illegal words");
        return Progress.DONE;
    }

    private Boolean containsOnlyAcceptedWords(FieldValue termValue){
        logger.info("  Checking if all words are accepted");
        if (!acceptedWords.isEmpty()){
            String query = termValue.toString().toLowerCase();
            String[] terms = query.split("\\s+");
            for (String term : terms) {
                if (!acceptedWords.contains(term)) {
                    return false;
                }
            }
        }
        return true;
    }

    private HashSet<String> getAcceptedWords(){
        logger.info("getting set of accepted words");
        HashSet<String> acceptedWords = new HashSet<String>();
        if (resourceExists()){
            try{
                ClassLoader cl = getClass().getClassLoader();
                InputStream is = cl.getResourceAsStream(filePath);
                InputStreamReader isr = new InputStreamReader(is, StandardCharsets.UTF_8);
                BufferedReader br = new BufferedReader(isr);
                String term;
                while ((term = br.readLine()) != null){
                    acceptedWords.add(term);
                }
            }catch (IOException e){
                e.printStackTrace();
            }
        }
        return acceptedWords;
    }

    private boolean resourceExists() {
        return getClass().getClassLoader().getResource(QueryDocumentProcessor.filePath) != null;
    }

    @Override
    public void deconstruct() {
        super.deconstruct();
    }
}