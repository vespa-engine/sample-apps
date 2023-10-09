// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site;

import ai.vespa.example.shopping.site.data.SimpleQueryBuilder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class SimpleQueryBuilderTest {

    @Test
    public void testQueryBuilding() {
        SimpleQueryBuilder query = new SimpleQueryBuilder("/search/");
        query.add("summary", "short");
        query.add("hits", "0");
        query.add("yql", "select * from sources item where default contains \"shoes\" | all( group(brand) each(output(count())) )\"");
        assertEquals("/search/?summary=short&hits=0&yql=select+*+from+sources+item+where+default+contains+%22shoes%22+%7C+all%28+group%28brand%29+each%28output%28count%28%29%29%29+%29%22", query.toString());
    }

}