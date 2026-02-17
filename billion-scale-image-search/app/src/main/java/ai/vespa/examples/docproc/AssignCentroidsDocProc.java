// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.docproc;

import ai.vespa.examples.Centroids;
import ai.vespa.examples.Centroids.CentroidResult;
import com.google.inject.Inject;
import com.yahoo.component.chain.Chain;
import com.yahoo.component.chain.dependencies.After;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DataType;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.Array;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.searchchain.ExecutionFactory;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;

import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.RejectedExecutionException;
import java.util.logging.Logger;

@After("DimensionReduction")
public class AssignCentroidsDocProc extends DocumentProcessor {

    private static final String PROMISE_VAR = AssignCentroidsDocProc.class.getName() + ".promise";
    private static final Logger logger = Logger.getLogger(AssignCentroidsDocProc.class.getName());
    private final Chain<Searcher> searchChain;
    private final ExecutionFactory factory;
    private final Centroids clustering;

    @Inject
    public AssignCentroidsDocProc(Centroids clusteringComponent, ExecutionFactory executionFactory) {
        this.clustering = clusteringComponent;
        this.factory = executionFactory;
        this.searchChain = executionFactory.searchChainRegistry().getChain("vespa");
    }

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut put) {
                Document doc = put.getDocument();
                if(doc.getDataType().getName().equalsIgnoreCase("centroid"))
                    return Progress.DONE;

                TensorFieldValue vector = (TensorFieldValue) doc.getFieldValue("reduced_vector");
                if(vector == null || vector.getTensor().isEmpty()) {
                    logger.warning("No `reduced_vector` found in document, failing operation");
                    return Progress.FAILED.withReason("No 'reduced_vector' tensor field in document");
                }

                // Check if neighbor search is already initiated
                var promise = getNeighborsSearchPromiseOrNull(processing);

                Duration timeout = timeout(processing);

                if (timeout.equals(Duration.ZERO)) return Progress.FAILED.withReason("Timed out");

                // Initiate neighbor search and return
                if (promise == null) {
                    findNeighborsAsync(processing, vector.getTensor().get().cellCast(TensorType.Value.FLOAT), timeout);
                    return later(timeout);
                }

                // Return if search not yet complete
                if (!promise.isDone()) return later(timeout);

                // Fail feed on search error
                CentroidResult result = Objects.requireNonNull(promise.getNow(null));
                if (result.isEmpty()) {
                    ErrorMessage error = result.getCentroidResult().hits().getError();
                    if(error != null)
                        logger.warning("Search for centroids failed with error " + error.getDetailedMessage());
                    return error != null
                            ? Progress.FAILED.withReason(error.getDetailedMessage())
                            : Progress.FAILED.withReason("Found no centroids");
                }

                // Cleanup `promise` variable since search completed successfully
                processing.removeVariable(PROMISE_VAR);

                // Assign near centroids from search result
                Array<StringFieldValue> clusters = new Array<>(DataType.getArray(DataType.STRING));
                for (var c : result.getCentroids())
                    clusters.add(new StringFieldValue(c.getId().toString()));
                doc.setFieldValue("centroids", clusters);
                return Progress.DONE;
            }
        }
        return Progress.DONE;
    }

    @SuppressWarnings("unchecked")
    private static CompletableFuture<CentroidResult> getNeighborsSearchPromiseOrNull(Processing p) {
        return (CompletableFuture<CentroidResult>) p.getVariable(PROMISE_VAR);
    }

    private static Duration timeout(Processing p) {
        Duration timeLeft = p.timeLeft();
        if (timeLeft == Processing.NO_TIMEOUT) return Duration.ofSeconds(60);
        if (timeLeft.toMillis() < 6) return Duration.ZERO;
        return timeLeft;
    }

    private static Progress later(Duration timeout) {
        return Progress.later(Math.min(20, timeout.minusMillis(2).toMillis()));
    }

    /**
     * Find neighbors asynchronously using the jdisc container's default thread pool
     */
    private void findNeighborsAsync(Processing p, Tensor t, Duration timeout) {
        try {
            Execution exec = factory.newExecution(searchChain);
            var promise = CompletableFuture.supplyAsync(
                    () -> clustering.getCentroids(t, 24, 48, timeout.minusMillis(4), exec),
                    exec.context().executor());
            p.setVariable(PROMISE_VAR, promise);
        } catch (RejectedExecutionException e) {
            // Ensure that search is retried later on back-pressure signal from search
            p.removeVariable(PROMISE_VAR);
        }
    }
}
