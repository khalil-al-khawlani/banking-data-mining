# Project Title: Banking Dataset Analysis
# Group Number: 1
# Student Names: (أضف أسماء الأعضاء هنا)

#### Full Data Mining Workflow (CRISP-DM) using R
#### This script is POSIT Cloud compatible. Place it at repository root and run in RStudio / Posit.

## ---------------------------
## 0. Setup: packages, options, paths
## ---------------------------
# Clear environment
rm(list = ls())
options(stringsAsFactors = FALSE)
set.seed(42)

# Required packages
required_pkgs <- c(
  'tidyverse', 'lubridate', 'caret', 'randomForest', 'rpart', 'rpart.plot',
  'factoextra', 'tidytext', 'textdata', 'wordcloud', 'knitr'
)

# Install missing packages (Posit Cloud allows install.packages)
inst <- required_pkgs[!(required_pkgs %in% installed.packages()[, 'Package'])]
if(length(inst)) install.packages(inst, repos='https://cloud.r-project.org')

# Load libraries
library(tidyverse)
library(lubridate)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(factoextra)
library(tidytext)
library(wordcloud)
library(knitr)

# Create output folders
if(!dir.exists('analysis/outputs')) dir.create('analysis/outputs', recursive = TRUE)
if(!dir.exists('analysis/plots')) dir.create('analysis/plots', recursive = TRUE)

data_path <- file.path('Project_datasets','banking_realistic.csv')
if(!file.exists(data_path)){
  stop('Dataset not found at ', data_path, '. Make sure file exists in this path.')
}

## ---------------------------
## 1. Problem Definition (CRISP-DM: Business understanding)
## ---------------------------
# Problem: Predict loan approval (LoanStatus) and analyze factors affecting loan decisions.
# Objective: Build classification model for `LoanStatus`, regression model for `Score`,
# and provide exploratory analysis, clustering and text (sentiment) analysis on `Review`.
# Expected outcome: Models (classification + regression), feature importance, visuals and recommendations.

## ---------------------------
## 2. Data Understanding
## ---------------------------
# Read data
df <- readr::read_csv(data_path, show_col_types = FALSE)

# Quick overview
cat('\n--- Data Understanding ---\n')
cat('File:', data_path, '\n')
cat('Rows:', nrow(df), 'Cols:', ncol(df), '\n')
cat('Column names:', paste(colnames(df), collapse=', '), '\n\n')

# Data types summary
str(df)

# Count missing values per column
na_counts <- sapply(df, function(x) sum(is.na(x)))
print(na_counts)

## ---------------------------
## 3. Data Cleaning
## ---------------------------
# Fix common issues: types, missing, duplicates

# Convert Date to Date if possible
if('Date' %in% names(df)){
  # Try common formats
  df <- df %>% mutate(Date = lubridate::ymd(Date))
}

# Ensure categorical types
df <- df %>% mutate(
  LoanStatus = as.factor(LoanStatus),
  Category = if('Category' %in% names(df)) as.factor(Category) else factor(NA),
  Review = if('Review' %in% names(df)) as.character(Review) else NA_character_
)

# Remove duplicate CustomerID rows if any (keep first)
if('CustomerID' %in% names(df)){
  dup_count <- sum(duplicated(df$CustomerID))
  cat('Duplicate CustomerID rows:', dup_count, '\n')
  df <- df %>% distinct(CustomerID, .keep_all = TRUE)
}

# Numeric columns we expect
num_vars <- intersect(c('Income','Expenditure','Score','Age'), names(df))

# Replace numeric NAs with median
for(v in num_vars){
  if(sum(is.na(df[[v]]))>0){
    med <- median(df[[v]], na.rm = TRUE)
    df[[v]][is.na(df[[v]])] <- med
    cat('Imputed NAs in', v, 'with median =', med, '\n')
  }
}

# Replace NA reviews with placeholder
if('Review' %in% names(df)){
  df$Review[is.na(df$Review) | trimws(df$Review)==""] <- 'NoReview'
}

# Confirm cleaning
cat('\nAfter cleaning: NA counts\n')
print(sapply(df, function(x) sum(is.na(x))))

## Save cleaned snapshot
readr::write_csv(df, file.path('analysis','outputs','banking_cleaned.csv'))

## ---------------------------
## 4. Data Exploration
## ---------------------------
cat('\n--- Data Exploration ---\n')

# Summary statistics for numeric vars
num_summary <- df %>% select(all_of(num_vars)) %>% summary()
print(num_summary)
knitr::kable(num_summary)

# ggplot distributions
if('Income' %in% names(df)){
  p1 <- ggplot(df, aes(x=Income)) + geom_histogram(bins=30, fill='#2c7fb8') + theme_minimal() + ggtitle('Income distribution')
  ggsave(filename='analysis/plots/income_hist.png', plot=p1, width=7, height=4)
}
if('Expenditure' %in% names(df)){
  p2 <- ggplot(df, aes(x=Expenditure)) + geom_histogram(bins=30, fill='#de2d26') + theme_minimal() + ggtitle('Expenditure distribution')
  ggsave(filename='analysis/plots/expenditure_hist.png', plot=p2, width=7, height=4)
}
if('Score' %in% names(df)){
  p3 <- ggplot(df, aes(x=Score)) + geom_histogram(bins=20, fill='#31a354') + theme_minimal() + ggtitle('Score distribution')
  ggsave(filename='analysis/plots/score_hist.png', plot=p3, width=7, height=4)
}

# Boxplots by LoanStatus
if(all(c('LoanStatus','Income') %in% names(df))){
  p4 <- ggplot(df, aes(x=LoanStatus, y=Income, fill=LoanStatus)) + geom_boxplot() + theme_minimal() + ggtitle('Income by LoanStatus')
  ggsave('analysis/plots/income_by_loanstatus.png', p4, width=7, height=4)
}
if(all(c('LoanStatus','Score') %in% names(df))){
  p5 <- ggplot(df, aes(x=LoanStatus, y=Score, fill=LoanStatus)) + geom_boxplot() + theme_minimal() + ggtitle('Score by LoanStatus')
  ggsave('analysis/plots/score_by_loanstatus.png', p5, width=7, height=4)
}

# Scatter Income vs Expenditure colored by LoanStatus
if(all(c('Income','Expenditure') %in% names(df))){
  p6 <- ggplot(df, aes(x=Income, y=Expenditure, color=LoanStatus)) + geom_point(alpha=0.6) + theme_minimal() + ggtitle('Income vs Expenditure')
  ggsave('analysis/plots/income_vs_expenditure.png', p6, width=7, height=4)
}

# Correlation matrix for numeric vars
if(length(num_vars) >= 2){
  cor_mat <- cor(df %>% select(all_of(num_vars)), use='complete.obs')
  write.csv(cor_mat, 'analysis/outputs/correlation_matrix.csv', row.names = TRUE)
  print(cor_mat)
}

## Identify outliers (simple rule: beyond 3*IQR)
outlier_report <- list()
for(v in num_vars){
  q <- quantile(df[[v]], probs=c(0.25,0.75), na.rm=TRUE)
  iqr <- q[2]-q[1]
  lower <- q[1] - 3*iqr
  upper <- q[2] + 3*iqr
  outlier_idx <- which(df[[v]] < lower | df[[v]] > upper)
  outlier_report[[v]] <- length(outlier_idx)
}
print(outlier_report)

## ---------------------------
## 5. Data Transformation (aggregation/grouping)
## ---------------------------
cat('\n--- Data Transformation / Aggregation ---\n')

# Derive month and day_of_week if Date exists
if('Date' %in% names(df) && !all(is.na(df$Date))){
  df <- df %>% mutate(
    date_new = as.Date(Date),
    day_of_week = lubridate::wday(date_new, label = TRUE, abbr = TRUE),
    month = lubridate::month(date_new, label = TRUE, abbr = TRUE)
  )

  monthly_summary <- df %>% group_by(month) %>% summarise(
    number_of_movements = n(),
    sum_of_entries = sum(Income, na.rm = TRUE),
    sum_of_expenses = sum(Expenditure, na.rm = TRUE)
  ) %>% arrange(month)

  write_csv(monthly_summary, 'analysis/outputs/monthly_summary.csv')
  print(monthly_summary)

  # Plots
  p_month_mov <- ggplot(monthly_summary, aes(x=month, y=number_of_movements)) + geom_col(fill='#3182bd') + coord_flip() + theme_minimal() + ggtitle('Number of movements per month')
  ggsave('analysis/plots/monthly_movements.png', p_month_mov, width=7, height=4)

  p_month_exp <- ggplot(monthly_summary, aes(x=month, y=sum_of_expenses)) + geom_col(fill='#de2d26') + coord_flip() + theme_minimal() + ggtitle('Sum of expenses per month')
  ggsave('analysis/plots/monthly_expenses.png', p_month_exp, width=7, height=4)

  p_month_inc <- ggplot(monthly_summary, aes(x=month, y=sum_of_entries)) + geom_col(fill='#31a354') + coord_flip() + theme_minimal() + ggtitle('Sum of income per month')
  ggsave('analysis/plots/monthly_income.png', p_month_inc, width=7, height=4)
}

## ---------------------------
## 6. Regression Analysis (predict numeric target: Score)
## ---------------------------
if('Score' %in% names(df) && length(num_vars) >= 2){
  cat('\n--- Regression: Predicting Score ---\n')

  # Prepare dataset: use numeric predictors only for linear regression
  reg_vars <- intersect(c('Score','Income','Expenditure','Age'), names(df))
  reg_df <- df %>% select(all_of(reg_vars)) %>% drop_na()

  # Train/test split
  set.seed(42)
  train_idx <- createDataPartition(reg_df$Score, p = 0.8, list = FALSE)
  train_reg <- reg_df[train_idx,]
  test_reg <- reg_df[-train_idx,]

  # Linear model
  lm_mod <- lm(Score ~ ., data = train_reg)
  summary_lm <- summary(lm_mod)
  print(summary_lm)

  # Predict and evaluate
  preds_lm <- predict(lm_mod, test_reg)
  rmse <- sqrt(mean((preds_lm - test_reg$Score)^2))
  cat('Linear Regression RMSE:', rmse, '\n')

  # Save model summary
  capture.output(summary_lm, file = 'analysis/outputs/linear_model_summary.txt')
}

## ---------------------------
## 7. Classification (predict LoanStatus)
## ---------------------------
if('LoanStatus' %in% names(df)){
  cat('\n--- Classification: Predicting LoanStatus ---\n')

  # Select features (numeric + Category)
  class_vars <- c('LoanStatus', intersect(c('Income','Expenditure','Score','Age','Category'), names(df)))
  class_df <- df %>% select(all_of(class_vars)) %>% drop_na()

  # Encode factors
  class_df <- class_df %>% mutate(LoanStatus = droplevels(LoanStatus))

  # Split
  set.seed(42)
  tr_idx <- createDataPartition(class_df$LoanStatus, p = 0.8, list = FALSE)
  train_cl <- class_df[tr_idx,]
  test_cl <- class_df[-tr_idx,]

  # Decision Tree (baseline classification model)
  tree_mod <- rpart(LoanStatus ~ . , data = train_cl, method = 'class')
  pred_tree <- predict(tree_mod, test_cl, type = 'class')
  cm_tree <- caret::confusionMatrix(pred_tree, test_cl$LoanStatus)
  print(cm_tree)
  capture.output(cm_tree, file = 'analysis/outputs/tree_confusion.txt')

  # Plot tree
  png('analysis/plots/decision_tree.png', width = 900, height = 700)
  rpart.plot(tree_mod, main = 'Decision Tree for LoanStatus')
  dev.off()

  # Random Forest
  rf_mod <- randomForest(LoanStatus ~ ., data = train_cl, ntree = 200)
  pred_rf <- predict(rf_mod, test_cl)
  cm_rf <- caret::confusionMatrix(pred_rf, test_cl$LoanStatus)
  print(cm_rf)
  capture.output(cm_rf, file = 'analysis/outputs/rf_confusion.txt')

  # Feature importance
  imp <- importance(rf_mod)
  imp_df <- data.frame(Feature = rownames(imp), Importance = imp[,1]) %>% arrange(desc(Importance))
  write_csv(imp_df, 'analysis/outputs/rf_feature_importance.csv')
  print(imp_df)

  # Plot variable importance
  p_imp <- ggplot(imp_df, aes(x=reorder(Feature, Importance), y=Importance)) + geom_col(fill='#2b8cbe') + coord_flip() + theme_minimal() + ggtitle('Random Forest Feature Importance')
  ggsave('analysis/plots/rf_feature_importance.png', p_imp, width=7, height=4)
}

## ---------------------------
## 8. Clustering (K-means on numeric vars)
## ---------------------------
if(length(num_vars) >= 2){
  cat('\n--- Clustering (K-means) ---\n')
  clust_df <- df %>% select(all_of(num_vars)) %>% drop_na()
  scaled <- scale(clust_df)

  # Choose k (simple heuristic: 2:6)
  wss <- map_dbl(2:6, function(k) {sum(kmeans(scaled, centers=k, nstart=10)$withinss)})
  k_choice <- which.min(wss) + 1 # not perfect, but we pick argmin
  cat('WSS for k=2..6:', wss, '\nChosen k (heuristic):', k_choice, '\n')

  km <- kmeans(scaled, centers = k_choice, nstart = 25)
  clust_plot <- fviz_cluster(km, data = scaled, geom = 'point', repel = TRUE) + ggtitle('K-means clusters')
  ggsave('analysis/plots/kmeans_clusters.png', clust_plot, width=7, height=5)

  # Table clusters vs LoanStatus if available
  if('LoanStatus' %in% names(df)){
    cluster_lookup <- df %>% select(LoanStatus) %>% drop_na()
    cluster_table <- table(km$cluster, cluster_lookup$LoanStatus)
    write.csv(as.data.frame(cluster_table), 'analysis/outputs/cluster_loanstatus_table.csv')
    print(cluster_table)
  }
}

## ---------------------------
## 9. Text Analysis (sentiment on Review)
## ---------------------------
if('Review' %in% names(df)){
  cat('\n--- Text Analysis / Sentiment ---\n')

  reviews <- df %>% select(CustomerID, LoanStatus, Review) %>% distinct(CustomerID, .keep_all = TRUE)
  tidy_rev <- reviews %>% tidytext::unnest_tokens(word, Review)

  bing <- get_sentiments('bing')
  sentiment_counts <- tidy_rev %>% inner_join(bing, by='word') %>% count(CustomerID, sentiment) %>% pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)

  # Merge back
  df_sent <- left_join(reviews, sentiment_counts, by = 'CustomerID')
  df_sent <- df_sent %>% mutate(positive = ifelse(is.na(positive), 0, positive), negative = ifelse(is.na(negative), 0, negative), sentiment_score = positive - negative)
  write_csv(df_sent, 'analysis/outputs/reviews_sentiment.csv')

  # Simple aggregate: average sentiment by LoanStatus
  if('LoanStatus' %in% names(df_sent)){
    sent_summary <- df_sent %>% group_by(LoanStatus) %>% summarise(mean_sent = mean(sentiment_score, na.rm=TRUE), n = n())
    print(sent_summary)
  }

  # Wordcloud of most common words (excluding stop words)
  data(stop_words)
  words <- tidy_rev %>% anti_join(stop_words, by='word') %>% count(word, sort=TRUE)
  if(nrow(words) == 0){
    words <- tibble(word = 'no_words', n = 1)
  }
  png('analysis/plots/review_wordcloud.png', width=800, height=600)
  wordcloud(words = words$word, freq = words$n, max.words = 100, colors = RColorBrewer::brewer.pal(8, 'Dark2'))
  dev.off()
}

## ---------------------------
## 10. Interpretation and Conclusion
## ---------------------------
cat('\n--- Interpretation & Conclusion ---\n')

cat('Files and plots produced in analysis/outputs and analysis/plots.\n')
cat('Summary: See feature importance (analysis/outputs/rf_feature_importance.csv), classification results (analysis/outputs/*_confusion.txt), and regression summary (analysis/outputs/linear_model_summary.txt)\n')

# Recommendations placeholder - fill after reviewing outputs
cat('\nKey findings and recommendations: (fill in after running the analysis)\n')

## End of script
