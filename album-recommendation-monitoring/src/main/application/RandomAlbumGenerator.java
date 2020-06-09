package application;

import json.object.*;
import json.object.ImmutableAlbum;
import json.object.ImmutableCategory;
import json.object.ImmutableCategory_Scores;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;


public class RandomAlbumGenerator {
    private final Category pop = ImmutableCategory.builder().cat("pop").build();
    private final Category rock = ImmutableCategory.builder().cat("rock").build();
    private final Category jazz = ImmutableCategory.builder().cat("jazz").build();
    private List<String> words;
    private List<String> names;
    private final Random random = new Random();

    public RandomAlbumGenerator() {
        try {
            String base_path = Paths.get("").toAbsolutePath() + "/src/main/lists/";
            words = Files.lines(Paths.get(base_path + "words_alpha.txt")).collect(Collectors.toList());
            names = Files.lines(Paths.get(base_path + "names.txt")).collect(Collectors.toList());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public TopLevel getRandomAlbum() {
        Cell score1 = json.object.ImmutableCell.builder()
                .address(pop)
                .value(random.nextDouble())
                .build();
        Cell score2 = json.object.ImmutableCell.builder()
                .address(rock)
                .value(random.nextDouble())
                .build();
        Cell score3 = json.object.ImmutableCell.builder()
                .address(jazz)
                .value(random.nextDouble())
                .build();
        Category_Scores category_scores = ImmutableCategory_Scores.builder()
                .cells(List.of(score1, score2, score3))
                .build();
        Album album = ImmutableAlbum.builder()
                .album(String.format("%s %s %s", words.get(random.nextInt(words.size())), words.get(random.nextInt(words.size())), words.get(random.nextInt(words.size()))))
                .artist(String.format("%s %s", names.get(random.nextInt(names.size())), names.get(random.nextInt(names.size()))))
                .year(((int) Math.floor(random.nextDouble() * 40)) + 1980)
                .category_scores(category_scores)
                .build();
        return ImmutableTopLevel.builder()
                .fields(album)
                .build();
    }
}
