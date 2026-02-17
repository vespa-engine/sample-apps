// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples;

import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.result.Hit;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.TensorType;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Utils {
    public static Result readTestResponse(Query query) throws FileNotFoundException, JSONException {
        Result result = new Result(query);
        String content = new Scanner(new File("src/test/resources/duplicate-test-data.json")).useDelimiter("\\Z").next();
        JSONArray images = new JSONArray(content);
        for (int i = 0; i < images.length(); i++) {
            JSONObject object = (JSONObject) images.get(i);
            String id = object.getString("id");
            double relevance= object.getDouble("relevance");
            JSONObject fields = object.getJSONObject("fields");
            String caption = fields.getString("caption");
            String url = fields.getString("url");
            JSONArray vector = fields.getJSONObject("vector").getJSONArray("values");
            TensorType type = new TensorType.Builder(TensorType.Value.FLOAT).indexed("d0", vector.length()).build();
            IndexedTensor.Builder builder = IndexedTensor.Builder.of(type);
            for (int j = 0; j <vector.length(); j++)
                builder.cell(vector.getFloat(j), j);

            Hit hit = new Hit(id,relevance);
            hit.setField("id", id);
            hit.setField("caption", caption);
            hit.setField("url", url);
            hit.setField("vector", builder.build());
            result.hits().add(hit);
        }
        result.setTotalHitCount(result.hits().size());
        return result;
    }
}
