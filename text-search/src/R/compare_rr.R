library(ggplot2)

data_folder <- "msmarco/"

rr_bm25 <- read.delim(file = file.path(data_folder, "test-output-bm25.tsv"), col.names = c("rr"))
rr_native <- read.delim(file = file.path(data_folder, "test-output-default.tsv"), col.names = c("rr"))

summary(rr_bm25)
summary(rr_native)

data <- rbind(data.frame(rr = rr_bm25, ranking = "bm25"), data.frame(rr = rr_native, ranking = "nativeRank"))

ggplot(data) + geom_boxplot(aes(x = ranking, y = rr))

data_folder <- "data/dev_test_data/"

rr_bm25 <- read.delim(file = file.path(data_folder, "text-search-tf-rank-docs-100_bm25_rr.tsv"), col.names = c("qid", "rr"))
rr_pointwise_bm25 <- read.delim(file = file.path(data_folder, "text-search-tf-rank-docs-100_pointwise_linear_bm25_rr.tsv"), col.names = c("qid", "rr"))
rr_listwise_bm25 <- read.delim(file = file.path(data_folder, "text-search-tf-rank-docs-100_listwise_linear_bm25_rr.tsv"), col.names = c("qid", "rr"))

data <- rbind(data.frame(rr = rr_pointwise_bm25$rr, model = "pointwise_bm25"), 
              data.frame(rr = rr_listwise_bm25$rr, model = "listwise_bm25"))

ggplot(data=subset(data, rr >= 1/10), aes(x=as.factor(1/rr), y = (..count..)/sum(..count..), fill=model)) +
  geom_bar(position=position_dodge()) + labs(x = "Position of the relevant document", y = "Frequency")

