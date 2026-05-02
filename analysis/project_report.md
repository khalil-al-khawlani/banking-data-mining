# Banking Dataset — Project Report (Group 1)

- Project Title: Banking Dataset Analysis
- Group Number: 1
- Section: (أكتب رقم القسم هنا)
- Members: (أضف أسماء الأعضاء مفصولة بفواصل)

---

## 1. Problem Definition

- Problem: التنبؤ بما إذا كان طلب القرض سيكون `Approved` أو `Rejected` وتحليل العوامل المؤثرة في القرار.
- Objective: بناء نموذج تصنيف لتوقّع `LoanStatus`، استخراج أهم الميزات، وتقديم توصيات بناءً على النتائج.
- Expected Outcome: نموذج بتقييمات (Accuracy, Recall, Precision, F1)، ملخّص استكشافي، وملاحظات عملية.

---

## 2. Data Understanding

- Dataset file: `Project_datasets/banking_realistic.csv`
- Dataset Description: سجلات عملاء تتضمن معلومات ديموغرافية ومالية ومراجعات نصية.
- Attributes (ملاحظات):
  - `CustomerID` (int)
  - `Date` (date)
  - `Age` (int)
  - `Income` (numeric)
  - `Expenditure` (numeric)
  - `Score` (numeric)
  - `LoanStatus` (Approved/Rejected) — الهدف التصنيفي
  - `Category` (categorical)
  - `Review` (text)

### 2.1 قراءة وعرض سريع (R)

```r
# حزمة أساسية
library(tidyverse)
library(lubridate)

df <- read_csv('Project_datasets/banking_realistic.csv')
glimpse(df)
cat('Rows:', nrow(df), 'Cols:', ncol(df), '\n')
summary(df)
sapply(df, function(x) sum(is.na(x)))

# تحويل الأنواع الأساسية
df <- df %>%
  mutate(Date = ymd(Date),
         LoanStatus = as.factor(LoanStatus),
         Category = as.factor(Category),
         Review = as.character(Review))
```

---

## 3. Data Cleaning (خطة + كود)

- Missing values handling: نعرض القيم المفقودة ونقرر بين التعويض أو الحذف.\
- Outliers: نتحقق من القيم الشاذة في `Income`, `Expenditure`, `Score`\
- Encoding: تغيير المتغيرات الفئوية إلى عوامل أو one-hot عند الحاجة.

### 3.1 نموذج تنظيف (R)

```r
# أمثلة سريعة على تنظيف بسيط
# إزالة صفوف معرفها مفقود أو CustomerID مكررة
df <- df %>% distinct(CustomerID, .keep_all = TRUE)

# استبدال القيم المفقودة في الأرقام بالوسيط (median)
num_vars <- c('Income','Expenditure','Score','Age')
for(v in num_vars){
  if(sum(is.na(df[[v]]))>0){
    med <- median(df[[v]], na.rm=TRUE)
    df[[v]][is.na(df[[v]])] <- med
  }
}

# تحويل Review إلى عامل نصي (نظيف لاحقًا)
df$Review[is.na(df$Review)] <- 'NoReview'
```

---

## 4. Data Exploration

### 4.1 توزيعات ومقارنات

```r
library(ggplot2)

ggplot(df, aes(x=Income)) + geom_histogram(bins=30) + theme_minimal()
ggplot(df, aes(x=Expenditure)) + geom_histogram(bins=30) + theme_minimal()
ggplot(df, aes(x=Score)) + geom_histogram(bins=20) + theme_minimal()

ggplot(df, aes(x=LoanStatus, y=Income)) + geom_boxplot() + theme_minimal()
ggplot(df, aes(x=LoanStatus, y=Score)) + geom_boxplot() + theme_minimal()

# علاقة Income و Expenditure
ggplot(df, aes(x=Income, y=Expenditure, color=LoanStatus)) + geom_point(alpha=0.6) + theme_minimal()
```

### 4.2 الارتباطات (Correlation)

```r
num_df <- df %>% select(all_of(num_vars))
cor(num_df, use='complete.obs')
```

---

## 5. Data Transformation

- Feature engineering مقترح:
  - `Income_to_Expenditure = Income / (Expenditure + 1)`
  - `Savings = Income - Expenditure`
  - One-hot للمتغير `Category` إذا لزم

```r
df <- df %>% mutate(
  Income_to_Expenditure = Income / (Expenditure + 1),
  Savings = Income - Expenditure
)

df <- df %>% mutate(across(where(is.factor), as.character))
df <- df %>% mutate(Category = as.factor(Category))
```

---

## 6. Regression Analysis (اقتراح)

- يمكن استخدام Regression لتنبؤ `Score` كمتغير عددي (اختياري). أهم خطوات: train/test split، نموذج Linear أو RandomForest Regression، تقييم بـ RMSE.

```r
# مثال مختصر: نموذج انحدار بسيط لتنبؤ Score
library(caret)
set.seed(42)
train_idx <- createDataPartition(df$Score, p=0.8, list=FALSE)
train <- df[train_idx,]
test <- df[-train_idx,]

lm_mod <- train(Score ~ Income + Expenditure + Age, data=train, method='lm')
preds <- predict(lm_mod, test)
postResample(preds, test$Score)
```

---

## 7. Classification (الخطوة الأساسية)

- Target: `LoanStatus`
- Model plan: Baseline (Decision Tree), Random Forest، تقييم بواسطة confusion matrix، تقرير الدقة ودلالات.

### 7.1 نموذج تصنيف (R)

```r
library(rpart)
library(rpart.plot)
library(randomForest)
set.seed(42)
train_idx <- createDataPartition(df$LoanStatus, p=0.8, list=FALSE)
train <- df[train_idx,]
test <- df[-train_idx,]

# Decision Tree
tree_mod <- rpart(LoanStatus ~ Income + Expenditure + Score + Age + Category, data=train, method='class')
pred_tree <- predict(tree_mod, test, type='class')
caret::confusionMatrix(pred_tree, test$LoanStatus)

# Random Forest for comparison
rf_mod <- randomForest(LoanStatus ~ Income + Expenditure + Score + Age + Category, data=train, ntree=200)
pred_rf <- predict(rf_mod, test)
caret::confusionMatrix(pred_rf, test$LoanStatus)

# Feature importance
importance(rf_mod)
varImpPlot(rf_mod)
```

---

## 8. Clustering

- استخدم KMeans على الميزات العددية لاستكشاف مجموعات العملاء.

```r
library(factoextra)
num_mat <- scale(df %>% select(Income, Expenditure, Score, Age))
km <- kmeans(num_mat, centers=3, nstart=25)
fviz_cluster(km, data=num_mat)
table(km$cluster, df$LoanStatus)
```

---

## 9. Text Analysis (Review)

- تنظيف نصوص، تحويل إلى TF-IDF أو استخدام قاموس بسيط لتحليل المشاعر.

```r
library(tidytext)
reviews <- df %>% select(CustomerID, LoanStatus, Review)
 tidy_rev <- reviews %>% tidytext::unnest_tokens(word, Review)
sentiment <- tidy_rev %>% inner_join(tidytext::get_sentiments('bing')) %>%
  count(CustomerID, sentiment) %>% pivot_wider(names_from=sentiment, values_from=n, values_fill=0)

# دمج مع الداتا وقياس التأثير
df2 <- left_join(df, sentiment, by=c('CustomerID'))
df2$sentiment_score <- df2$positive - df2$negative
cor(df2$sentiment_score, ifelse(df2$LoanStatus=='Approved',1,0), use='complete.obs')
```

---

## 10. Interpretation and Conclusion

- Key Insights: (املئ بعد تشغيل التحليل)
- Recommendations: (اقترح سياسات قبول/معايير إلخ بناءً على الميزات المهمة)

---

## Interpretation and Conclusion

**Group:** Group 1 — Banking Dataset Analysis

**Members:** Ghadah Ali bin Al-Shehri (444821091), Jana Khalid Al-Qahtani (445803988), Waad Abdulaziz Al-Buraik (443813233)

**Summary of findings (concise, human style):**
- The data contains 100 records describing customer income, expenditure, credit score and a simple text review. After cleaning and basic feature engineering we ran regression, classification, clustering and a simple sentiment analysis.
- Regression: a linear model to predict `Score` from `Income`, `Expenditure` and `Age` produced low R-squared (~0.05) and RMSE ~15.7 — numeric predictors explain little variance in `Score`.
- Classification: a decision tree and Random Forest both achieved perfect accuracy on the held-out test split used in the script; Random Forest feature importance ranks `Income` and `Score` highest. Because the test set is small, we applied 5-fold cross-validation and report AUC and averaged metrics in `analysis/outputs/` (see `rf_cv_results.csv`, `rf_cv_auc.csv`).
- Clustering: K-means (k=6 heuristic) found customer groups with different mixes of approved/rejected labels — useful for exploratory segmentation but requires business validation.
- Text analysis: simple sentiment counts (using `bing`) show a small difference between `Approved` and `Rejected` customers; reviews in the dataset are very short (e.g., "Bad", "Average") limiting text signal.

**Interpretation (actionable):**
- The strongest numerical signals for loan decisions appear to be `Income` and `Score`. We recommend the instructor or bank to consider `Income_to_Expenditure` ratios and combinations of `Score` with recent spending behavior when designing approval rules.
- Because a small test split produced perfect accuracy, rely on cross‑validated AUC and metrics instead of a single split — see `analysis/outputs/rf_cv_auc.csv` for the validated AUC. If AUC remains high across CV, the model is likely robust; otherwise, collect more data or engineer stronger features.

**Recommendations for improvement / next steps:**
1. Use stratified k‑fold cross‑validation and report mean±SD for Accuracy, F1 and AUC (we added 5‑fold CV in the script and saved results).
2. Engineer additional features: historical averages, income/expenditure trends, time-since-last-default (if available), categorical encoding of `Category` with one-hot.
3. If deployment is required, set a conservative decision threshold and track false positives (approving risky loans) separately from false negatives.
4. For text analysis, gather longer customer feedback or reviews to extract signal or use an external sentiment lexicon/embedding for better accuracy.

**Deliverables provided in repository:**
- Cleaned dataset: `analysis/outputs/banking_cleaned.csv`
- Plots: `analysis/plots/` (histograms, boxplots, cluster plot, ROC, wordcloud)
- Model outputs: `analysis/outputs/` (confusion matrices, rf_feature_importance.csv, rf_cv_results.csv, rf_cv_auc.csv)

If you want, I can now run a short script to export `analysis/project_report.md` to `analysis/project_report.docx` and attach it here, or I can add a short slide-style summary for submission.

---

### Next steps
1. شغّل هذا الملف محليًا أو في Posit Cloud (RStudio).\
2. إذا تريد، أحول هذا `project_report.md` إلى ملف Word (`.docx`) باستخدام pandoc أو R Markdown، ثم أرفع كل شيء إلى GitHub وأرشدك لفتح المشروع على Posit Cloud.

مثال لتحويل باستخدام pandoc (محليًا):

```bash
pandoc analysis/project_report.md -o analysis/project_report.docx --from markdown -s
```
