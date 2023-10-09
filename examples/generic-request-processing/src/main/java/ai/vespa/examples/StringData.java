// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples;

import com.yahoo.processing.Request;
import com.yahoo.processing.response.AbstractData;

/**
 * Example data which can be added to a Response.
 */
public class StringData extends AbstractData {

    private final String string;

    public StringData(Request request, String string) {
        super(request);
        this.string = string;
    }

    @Override
    public String toString() {
        return string;
    }

}
