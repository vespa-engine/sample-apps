package com.mydomain.example;

import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.StringFieldValue;

public class Rot13DocumentProcessor extends DocumentProcessor {
    private static final String FIELD_NAME = "title";

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document document = put.getDocument();

                StringFieldValue oldTitle = (StringFieldValue) document.getFieldValue(FIELD_NAME);
                if (oldTitle != null) {
                    document.setFieldValue(FIELD_NAME, rot13(oldTitle.getString()));
                }
            }
        }
        return DocumentProcessor.Progress.DONE;
    }

    private static String rot13(String s) {
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= 'a' && c <= 'm' || c >= 'A' && c <= 'M') {
                c += 13;
            } else if (c >= 'n' && c <= 'z' || c >= 'N' && c <= 'Z') {
                c -= 13;
            }
            output.append(c);
        }
        return output.toString();
    }
}