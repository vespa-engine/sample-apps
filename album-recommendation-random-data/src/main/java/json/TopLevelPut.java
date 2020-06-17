package json;

import org.immutables.gson.Gson;
import org.immutables.value.Value;

@Gson.TypeAdapters
@Value.Immutable
public abstract class TopLevelPut {
    public abstract String put();

    public abstract Album fields();
}
