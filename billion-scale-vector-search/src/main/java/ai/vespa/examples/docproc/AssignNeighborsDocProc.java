// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.docproc;

import ai.vespa.examples.ClusteringComponent;
import ai.vespa.examples.ClusteringComponent.CentroidResult;
import com.google.inject.Inject;
import com.yahoo.component.chain.Chain;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.BoolFieldValue;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.document.datatypes.WeightedSet;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.searchchain.ExecutionFactory;
import com.yahoo.tensor.Tensor;

import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.RejectedExecutionException;

public class AssignNeighborsDocProc extends DocumentProcessor {

    private static final String PROMISE_VAR = AssignNeighborsDocProc.class.getName() + ".promise";

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
                    return Progress.FAILED.withReason("No 'vector' tensor field in document");
                }

                BoolFieldValue inGraph = (BoolFieldValue)doc.getFieldValue("in_graph");
                if(inGraph == null)
                    return Progress.FAILED.withReason("No 'in_graph' field ");

                if(inGraph.getBoolean()) {
                    //Just forward to content cluster selection which
                    //will route the doc to the graph content cluster
                    return Progress.DONE;
                }

                // Check if neighbor search is already initiated
                var promise = getNeighborsSearchPromiseOrNull(processing);

                // Initiate neighbor search and return
                if (promise == null) return findNeighborsAsync(processing, vector);

                // Return if search not yet complete
                if (!promise.isDone()) return Progress.LATER;

                // Fail feed on search error
                CentroidResult result = Objects.requireNonNull(promise.getNow(null));
                if (result.isEmpty()) {
                    ErrorMessage error = result.getCentroidResult().hits().getError();
                    return error != null
                            ? Progress.FAILED.withReason(error.getMessage())
                            : Progress.FAILED.withReason("Found no neighbors");
                }

                // Cleanup `promise` variable since search completed successfully
                processing.removeVariable(PROMISE_VAR);

                // Assign `neighbors` from search result
                WeightedSet<StringFieldValue> clusters = new WeightedSet<>(DataType.getWeightedSet(DataType.STRING));
                for(ClusteringComponent.Centroid c: result.getCentroids())
                    clusters.put(new StringFieldValue(c.getId().toString()),c.getIntCloseness());
                doc.setFieldValue("neighbors", clusters);

                // Clear vector and replace with `disk_vector`
                Tensor vectorTensor = vector.getTensor().get();
                doc.setFieldValue("disk_vector", new TensorFieldValue(vectorTensor));
                vector.clear();
                return Progress.DONE;
            }
        }
        return Progress.DONE;
    }

    @SuppressWarnings("unchecked")
    private static CompletableFuture<CentroidResult> getNeighborsSearchPromiseOrNull(Processing p) {
        return (CompletableFuture<CentroidResult>) p.getVariable(PROMISE_VAR);
    }

    /**
     * Find neighbors asynchronously using the jdisc container's default thread pool
     */
    private Progress findNeighborsAsync(Processing p, TensorFieldValue t) {
        try {
            Execution exec = factory.newExecution(searchChain);
            Duration timeout = Duration.ofSeconds(30);
            var promise = CompletableFuture.supplyAsync(
                    () -> clustering.getCentroids(t.getTensor().get(), 12, 36, timeout, exec),
                    exec.context().executor());
            p.setVariable(PROMISE_VAR, promise);
        } catch (RejectedExecutionException e) {
            // Ensure that search is retried later on back-pressure signal from search
            p.removeVariable(PROMISE_VAR);
        }
        return Progress.LATER;
    }
}
