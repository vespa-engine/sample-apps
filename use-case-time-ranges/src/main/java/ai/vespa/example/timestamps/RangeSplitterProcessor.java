// Copyright Vespa.ai. All rights reserved.
package ai.vespa.example.timestamps;

import com.yahoo.docproc.SimpleDocumentProcessor;
import com.yahoo.document.ArrayDataType;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.Array;
import com.yahoo.document.datatypes.LongFieldValue;
import com.yahoo.document.datatypes.StringFieldValue;

import java.util.*;
import ai.vespa.example.timestamps.RangesplitterConfig;

public class RangeSplitterProcessor extends SimpleDocumentProcessor {

    public record FieldPair(String originalField, String rangeMarkerField) {
        @Override
        public String toString() {
            return originalField + " -> " + rangeMarkerField;
        }
    }
    private final List<FieldPair> fieldPairs;

    public RangeSplitterProcessor(RangesplitterConfig config) {
        this.fieldPairs = new ArrayList<>();
        for (RangesplitterConfig.Splitfields splitfield : config.splitfields()) {
            fieldPairs.add(new FieldPair(splitfield.originalField(), splitfield.rangeMarkerField()));
        }
    }

    @Override
    public void process(DocumentPut put) {
        Document document = put.getDocument();
        for (var fieldpair : fieldPairs) {
            var timestampField = document.getFieldValue(fieldpair.originalField());
            if (timestampField instanceof LongFieldValue lfv) {
                long theValue = lfv.getLong();
                Array newValue = new Array(DataType.getArray(DataType.STRING));
                for (String mark : RangeSplit.generateRangeMarks(theValue)) {
                    newValue.add(new StringFieldValue(mark));
                }
                document.setFieldValue(fieldpair.rangeMarkerField(), newValue);
            }
        }
        // System.out.println("PROCESSED: " + put.getId());
    }

}
