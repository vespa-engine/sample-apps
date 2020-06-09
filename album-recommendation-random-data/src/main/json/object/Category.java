package json.object;

import org.immutables.gson.Gson;
import org.immutables.value.Value;

import java.util.Random;
@Gson.TypeAdapters
@Value.Immutable
public abstract class Category {
    public abstract String cat();
}
