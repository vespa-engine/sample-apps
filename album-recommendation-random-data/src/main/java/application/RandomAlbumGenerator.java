package application;

import json.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
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
        words = new BufferedReader(new InputStreamReader(getClass().getClassLoader().getResourceAsStream("lists/words_alpha.txt"))).lines().collect(Collectors.toList());
        names = new BufferedReader(new InputStreamReader(getClass().getClassLoader().getResourceAsStream("lists/names.txt"))).lines().collect(Collectors.toList());
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
        Category_Scores category_scores = ImmutableCategory_Scores.builder()
                .cells(List.of(score1, score2, score3))
                .build();
        return ImmutableAlbum.builder()
                .album(String.format("%s %s %s", words.get(random.nextInt(words.size())), words.get(random.nextInt(words.size())), words.get(random.nextInt(words.size()))))
                .artist(String.format("%s %s", names.get(random.nextInt(names.size())), names.get(random.nextInt(names.size()))))
                .year(((int) Math.floor(random.nextDouble() * 40)) + 1980)
                .category_scores(category_scores)
                .build();
    }

    public static void main(String[] args) {
        RandomAlbumGenerator rando = new RandomAlbumGenerator();
        System.out.println("test");
    }
}
