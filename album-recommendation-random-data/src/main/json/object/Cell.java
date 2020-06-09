package json.object;

import org.immutables.gson.Gson;
import org.immutables.value.Value;

@Gson.TypeAdapters
@Value.Immutable
public abstract class Cell {
    public abstract Category address();
    public abstract Double value();
}

