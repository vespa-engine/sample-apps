library(ggplot2)

data_folder <- "data/dev_test_data/"

rr_bm25 <- read.delim(file = file.path(data_folder, "text-search-tf-rank-docs-100_bm25_rr.tsv"), col.names = c("qid", "rr"))
rr_pointwise_bm25 <- read.delim(file = file.path(data_folder, "text-search-tf-rank-docs-100_pointwise_linear_bm25_rr.tsv"), col.names = c("qid", "rr"))
rr_listwise_bm25 <- read.delim(file = file.path(data_folder, "text-search-tf-rank-docs-100_listwise_linear_bm25_rr.tsv"), col.names = c("qid", "rr"))

data <- rbind(data.frame(rr = rr_pointwise_bm25$rr, model = "pointwise_bm25"), 
              data.frame(rr = rr_listwise_bm25$rr, model = "listwise_bm25"))

ggplot(data=subset(data, rr >= 1/10), aes(x=as.factor(1/rr), y = (..count..)/sum(..count..), fill=model)) +
  geom_bar(position=position_dodge()) + labs(x = "Position of the relevant document", y = "Frequency") + theme(legend.position="top")

