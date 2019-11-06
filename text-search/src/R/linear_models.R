library(dplyr)
library(ggplot2)
library(reshape2)
library(arm)

#
# Load data
#
data <- read.csv("data/training_data_collectAll_match_random.csv")
drops <- c("docid", 
           "qid", 
           "relevant", 
           "rankingExpression.averagefieldTermMatchBodyFirstPosition.", 
           "rankingExpression.averagefieldTermMatchTitleFirstPosition.",
           "rankingExpression.maxfieldTermMatchBodyFirstPosition.",
           "rankingExpression.maxfieldTermMatchTitleFirstPosition.",
           "rankingExpression.averagefieldTermMatchBodyOcurrences.",
           "rankingExpression.maxfieldTermMatchBodyOcurrences.",
           "rankingExpression.maxfieldTermMatchTitleOcurrences.",
           "rankingExpression.averagefieldTermMatchTitleOcurrences.",
           "elementCompleteness.body..elementWeight",
           "elementCompleteness.title..elementWeight",
           "matches.body.",
           "matches.title."
)


#
# Filter queries that would not have been selected by the query
#
invalid_queries <- subset(data, bm25.body. == 0 & bm25.title. == 0) %>%
  select(qid)
valid_data <- subset(data, !(qid %in% unique(invalid_queries$qid)))

#
# Sample dataset and create train and validation data
#
set.seed(42)
valid_qids <- unique(valid_data$qid)
number_qid_to_sample <- length(valid_qids)
sample_qids <- sample(valid_qids, size = number_qid_to_sample, replace = FALSE)
train_idx <- sample(1:length(sample_qids), floor(length(sample_qids)/2), replace = FALSE)
sample_train_qids <- sample_qids[train_idx]
sample_val_qids <- sample_qids[-train_idx]

sample_train_data <- subset(valid_data, qid %in% sample_train_qids)
sample_train_data <- sample_train_data[!duplicated(sample_train_data), ]

sample_val_data <- subset(valid_data, qid %in% sample_val_qids)
sample_val_data <- sample_val_data[!duplicated(sample_val_data), ]

#
# Create function for reciprocal rank
#
rr <- function(observed, predicted){
  sorted_observed <- observed[order(predicted, decreasing = TRUE)]
  position_relevant <- match(1, sorted_observed)
  return(1/position_relevant)
}

#
# Create function to compute validation metrics
#
compute_validation_metrics <- function(validation_data, model){
  unique_queries <- unique(validation_data$qid)
  rr_values <- vector(mode = "numeric", length = length(unique_queries))
  count <- 0
  for (query_id in unique_queries){
    print(query_id)
    count <- count + 1
    x <- subset(validation_data, qid == query_id)
    predicted <- as.numeric(predict(model, newdata = x))
    observed <- x$relevant
    rr_values[count] <- rr(observed = observed, predicted = predicted)
  }
  return(rr_values)  
}

#
# Fit and compute evaluation metric
#
r1 = glm(formula = relevant ~ bm25.body. + bm25.title., family = binomial(), data = sample_train_data)
r1
r1_rr_values <- compute_validation_metrics(validation_data = sample_val_data, model=r1)

r5 = glm(formula = relevant ~ bm25.body. + bm25.title. + nativeRank.body. + nativeRank.title., family = binomial(), data = sample_train_data)
r5
r5_rr_values <- compute_validation_metrics(validation_data = sample_val_data, model=r5)

summary(r1_rr_values)
summary(r5_rr_values)