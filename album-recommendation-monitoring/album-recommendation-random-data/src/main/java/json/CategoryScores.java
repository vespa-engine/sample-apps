// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package json;

import org.immutables.gson.Gson;
import org.immutables.value.Value;

import java.util.List;

@Gson.TypeAdapters
@Value.Immutable
public abstract class CategoryScores {
    public abstract List<Cell> cells();
}
