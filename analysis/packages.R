## helper script to install required R packages for the notebook
required <- c('tidyverse', 'lubridate', 'caret', 'randomForest', 'rmarkdown', 'rpart', 'rpart.plot', 'factoextra', 'tidytext', 'textdata', 'wordcloud', 'knitr')
inst <- required[!required %in% installed.packages()[,'Package']]
if(length(inst)) install.packages(inst, repos='https://cloud.r-project.org')
message('Packages ready: ', paste(required, collapse=', '))
pandoc analysis/project_report.md -o analysis/project_report.docx --from markdown -s