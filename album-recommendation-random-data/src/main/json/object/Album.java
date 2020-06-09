package json.object;

import org.immutables.gson.Gson;
import org.immutables.value.Value;

import java.util.List;
@Gson.TypeAdapters
@Value.Immutable
public abstract class Album {
    public abstract String album();
    public abstract String artist();
    public abstract Integer year();
    public abstract Category_Scores category_scores();
}

