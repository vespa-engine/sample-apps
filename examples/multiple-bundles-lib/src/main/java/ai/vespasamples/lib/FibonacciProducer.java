// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespasamples.lib;

import java.util.logging.Logger;

public class FibonacciProducer {
    private static final Logger log = Logger.getLogger(FibonacciProducer.class.getName());

    private long n = 0;
    private long f = 1;
    private long f_1 = 1;

    private boolean stopped = false;

    public long getNext() {
        long ret = f;
        if (! stopped) {
            n++;
            if (n == 1) return 0;
            if (n == 2) return 1;

            long f_2 = f_1;
            f_1 = f;
            f = f_1 + f_2;
        }
        return ret;
    }

    public void stop() {
        log.info("The rabbits are going to sleep");
        stopped = true;
    }

}
