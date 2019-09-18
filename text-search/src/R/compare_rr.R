library(ggplot2)

setwd("~/projects/vespa/sw/msmarco-app/")

data_folder <- "data/sample_3000/"

rr_bm25 <- read.delim(file = file.path(data_folder, "test-output-bm25.tsv"), col.names = c("rr"))
rr_native <- read.delim(file = file.path(data_folder, "test-output-default.tsv"), col.names = c("rr"))

summary(rr_bm25)
summary(rr_native)

data <- rbind(data.frame(rr = rr_bm25, ranking = "bm25"), data.frame(rr = rr_native, ranking = "nativeRank"))

ggplot(data) + geom_histogram(aes(x = rr)) + facet_grid(. ~ ranking)
