// Copyright 2019 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.mydomain.lib;

import java.util.logging.Logger;

public class FibonacciProducer {
    private static final Logger log = Logger.getLogger(FibonacciProducer.class.getName());

    private long n = 0;
    private long f = 1;
    private long f_1 = 1;
    private long f_2 = 0;

    private boolean stopped = false;

    public long getNext() {
        if (n == 0) return 0;
        if (n == 1) return 1;

        long ret = f;
        if (! stopped) {
            f_2 = f_1;
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
