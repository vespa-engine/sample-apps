library(dplyr)
library(ggplot2)
library(reshape2)
library(arm)

#
# Load data
#
valid_data <- read.csv("data/training_data_match_random_collectAll.csv")

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
# Fit and compute evaluation metric for linear model
#
r1 = glm(formula = relevant ~ bm25.body. + bm25.title., family = binomial(), data = sample_train_data)
r1
r1_rr_values <- compute_validation_metrics(validation_data = sample_val_data, model=r1)

#
# Fit and compute evaluation metric for default bm25 linear model
#
r2 <- r1
r2$coefficients[1] <- 0
r2$coefficients[2] <- 1
r2$coefficients[3] <- 1
r2
r2_rr_values <- compute_validation_metrics(validation_data = sample_val_data, model=r2)

summary(r1_rr_values)
summary(r2_rr_values)

#
# Evaluation plots
#
data_to_plot <- rbind(data.frame(rr = r2_rr_values, model = "bm25"),
                      data.frame(rr = r1_rr_values, model = "linear_bm25"))

ggplot(data=data_to_plot, aes(x=as.factor(1/rr), y = (..count..)/sum(..count..), fill=model)) +
  geom_bar(position=position_dodge()) + labs(x = "Position of the relevant document", y = "Frequency")

#
# Test data plots
#
bm25_rr_test_values <- read.csv("data/dev_test_data/text-search-refeed-2_bm25_rr.tsv", 
                                sep = "\t", header = FALSE, col.names = c("qid", "rr"))
linear_bm25_rr_test_values <- read.csv("data/dev_test_data/text-search-refeed-2_linear_bm25_random_rr.tsv", 
                                       sep = "\t", header = FALSE, col.names = c("qid", "rr"))

rr_min_value <- min(subset(bm25_rr_test_values, rr > 0)$rr, subset(linear_bm25_rr_test_values, rr > 0)$rr)
bm25_rr_test_values[bm25_rr_test_values$rr == 0, ] <- rr_min_value
linear_bm25_rr_test_values[linear_bm25_rr_test_values$rr == 0, ] <- rr_min_value

test_data_to_plot <- rbind(data.frame(rr = bm25_rr_test_values$rr, model = "bm25"), 
                      data.frame(rr = linear_bm25_rr_test_values$rr, model = "linear_bm25"))


ggplot(data=test_data_to_plot, aes(x=cut(1/rr, c(0,5,10,15,20,50, 100, 500, 1000)), y = (..count..)/sum(..count..), fill=model)) +
  geom_bar(position=position_dodge()) + labs(x = "Position of the relevant document", y = "Frequency")

ggplot(data=subset(test_data_to_plot, rr >= 1/11), aes(x=as.factor(1/rr), y = (..count..)/sum(..count..), fill=model)) +
  geom_bar(position=position_dodge()) + labs(x = "Position of the relevant document", y = "Frequency")

