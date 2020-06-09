package json.object;

import org.immutables.gson.Gson;
import org.immutables.value.Value;

@Gson.TypeAdapters
@Value.Immutable
public abstract class TopLevel {
    public abstract Album fields();
}
