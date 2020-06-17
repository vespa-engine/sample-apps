package json;

import org.immutables.gson.Gson;
import org.immutables.value.Value;

@Gson.TypeAdapters
@Value.Immutable
public abstract class Album {
    public abstract String album();

    public abstract String artist();

    public abstract Integer year();

    public abstract CategoryScores category_scores();
}

