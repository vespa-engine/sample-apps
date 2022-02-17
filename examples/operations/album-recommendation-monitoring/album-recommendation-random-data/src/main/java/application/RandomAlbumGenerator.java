// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package application;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.stream.Collectors;
import json.Album;
import json.Category;
import json.CategoryScores;
import json.Cell;
import json.ImmutableAlbum;
import json.ImmutableCategory;
import json.ImmutableCategoryScores;
import json.ImmutableCell;


public class RandomAlbumGenerator {
    private final Category pop = ImmutableCategory.builder().cat("pop").build();
    private final Category rock = ImmutableCategory.builder().cat("rock").build();
    private final Category jazz = ImmutableCategory.builder().cat("jazz").build();
    private final Random random = new Random();
    private final List<String> words;
    private final List<String> names;

    public RandomAlbumGenerator() {
        words = new BufferedReader(new InputStreamReader(Objects.requireNonNull(getClass().getClassLoader().getResourceAsStream("lists/words_alpha.txt")))).lines().collect(Collectors.toList());
        names = new BufferedReader(new InputStreamReader(Objects.requireNonNull(getClass().getClassLoader().getResourceAsStream("lists/names.txt")))).lines().collect(Collectors.toList());
    }

    public Album getRandomAlbum() {
        Cell score1 = ImmutableCell.builder()
                .address(pop)
                .value(random.nextDouble())
                .build();
        Cell score2 = ImmutableCell.builder()
                .address(rock)
                .value(random.nextDouble())
                .build();
        Cell score3 = ImmutableCell.builder()
                .address(jazz)
                .value(random.nextDouble())
                .build();
        CategoryScores categoryScores = ImmutableCategoryScores.builder()
                .cells(List.of(score1, score2, score3))
                .build();
        return ImmutableAlbum.builder()
                .album(String.format("%s %s %s", words.get(random.nextInt(words.size())), words.get(random.nextInt(words.size())), words.get(random.nextInt(words.size()))))
                .artist(String.format("%s %s", names.get(random.nextInt(names.size())), names.get(random.nextInt(names.size()))))
                .year(random.nextInt(40) + 1980)
                .category_scores(categoryScores)
                .build();
    }
}
