library(ggplot2)

data_folder <- "msmarco/"

rr_bm25 <- read.delim(file = file.path(data_folder, "test-output-bm25.tsv"), col.names = c("rr"))
rr_native <- read.delim(file = file.path(data_folder, "test-output-default.tsv"), col.names = c("rr"))

summary(rr_bm25)
summary(rr_native)

data <- rbind(data.frame(rr = rr_bm25, ranking = "bm25"), data.frame(rr = rr_native, ranking = "nativeRank"))

ggplot(data) + geom_boxplot(aes(x = ranking, y = rr))