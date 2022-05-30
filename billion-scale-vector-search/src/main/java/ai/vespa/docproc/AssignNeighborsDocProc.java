// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.docproc;

import ai.vespa.ClusteringComponent;
import com.google.inject.Inject;
import com.yahoo.component.chain.Chain;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.*;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.searchchain.ExecutionFactory;
import com.yahoo.tensor.Tensor;

public class AssignNeighborsDocProc extends DocumentProcessor {

    private final Chain<Searcher> searchChain;
    private final ExecutionFactory factory;
    private final ClusteringComponent clustering;

    @Inject
    public AssignNeighborsDocProc(ClusteringComponent clusteringComponent, ExecutionFactory executionFactory) {
        this.clustering = clusteringComponent;
        this.factory = executionFactory;
        this.searchChain = executionFactory.searchChainRegistry().getChain("vespa");
    }

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document doc = put.getDocument();

                TensorFieldValue vector = (TensorFieldValue) doc.getFieldValue("vector");
                if(vector.getTensor().isEmpty()) {
                    return Progress.PERMANENT_FAILURE.withReason("No 'vector' tensor field in document");
                }

                BoolFieldValue inGraph = (BoolFieldValue)doc.getFieldValue("in_graph");
                if(inGraph == null)
                    return Progress.PERMANENT_FAILURE.withReason("No 'in_graph' field ");

                if(inGraph.getBoolean()) {
                    //Just forward to content cluster selection which
                    //will route the doc to the graph content cluster
                    return Progress.DONE;
                }
                //clear vector and add `disk_vector`
                Tensor vectorTensor = vector.getTensor().get();
                doc.setFieldValue("disk_vector", new TensorFieldValue(vectorTensor));
                vector.clear();
                return assignNeighbors(vectorTensor, doc);
            }
        }
        return Progress.DONE;
    }

    private Progress assignNeighbors(Tensor tensor, Document doc) {
        Execution execution = this.factory.newExecution(this.searchChain);
        ClusteringComponent.CentroidResult result = clustering.getCentroids(tensor, 12,36, execution);
        if (result.isEmpty()) {
            return Progress.LATER;
        }
        WeightedSet<StringFieldValue> clusters = new WeightedSet<>(DataType.getWeightedSet(DataType.STRING));
        for(ClusteringComponent.Centroid c: result.getCentroids())
            clusters.put(new StringFieldValue(c.getId().toString()),c.getIntCloseness());
        doc.setFieldValue("neighbors",clusters);
        return Progress.DONE;
    }
}
