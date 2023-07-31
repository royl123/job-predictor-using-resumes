# CST 383: Final Project Report #
**CST 383-30: Summer 2023**  
**Team Members: Emma George, Roy Luengas, Kevin Wical, Shawn Khalighi**

## Introduction ##
In today's fast-paced job market, efficiently categorizing and analyzing resumes has become crucial for job seekers and employers. With the ever-increasing number of applicants and the need to streamline the recruitment process, there is a growing demand for automated systems that can accurately predict the job category based on the contents of a resume. This report addresses this challenge by presenting a machine-learning system modeled to predict job categories using specific features extracted from resumes. 

## Selection of Data ##
We chose to work with a resumé dataset from Kaggle that included over 2400 resumés in twenty-four different categories. This dataset originally came with four columns, including a unique ID number, their full resumes in both string and HTML format, and their job category. There were two resumes that had duplicate entries and we dropped those rows in order to maintain uniqueness. In order to process the entries through some text preprocessing methods we ‘cleaned’ the resumes by adding a column that removed any non-alphanumeric characters. 

## Data Exploration ## 

**Bar Graph: Number of Resumes per Category**
![Number of resumes per category bar graph](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/number_resumes_per_category.png)

When looking at the job titles, we noticed some interesting patterns that could indicate bias. The 'AVIATION' category had the most unique job titles, showing a wide range of options. 'APPAREL' and 'HEALTHCARE' categories also had a good variety of job titles. On the other hand, the 'BPO' category had the fewest unique job titles. To understand bias better, it's important to look at the specific job titles in each category and see if there are any differences in representation for different groups. This analysis gives us a starting point to dig deeper into potential biases in job labels.

**Bar Graph: Number of Unique Job Titles per Category**
![Number of Unique Job Titles per Categor bar graph](https://github.com/emmariegeo/cst383final/blob/7f2b132d99c802f688370b7b79ef1263697c93b6/report_images/number_unique_jobtitles_per_category.png)

**10 Most Frequently Used Words by Job Category**
![HR Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/1-hr.png)  
![Designer Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/2-designer.png)
![IT Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/3-it.png)
![Teacher Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/4-teacher.png)
![Advocate Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/5-advocate.png)
![Business Development Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/6-businessdevelopment.png)
![Healthcare Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/7-healthcare.png)
![Fitness Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/8-fitness.png)
![Agriculture Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/9-ag.png)
![BPO Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/10-bpo.png)
![Sales Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/11-sales.png)
![Consultant Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/12-consultant.png)
![Digital Media Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/13-digitalmedia.png)
![Automotive Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/14-auto.png)
![Aviation Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/14-aviation.png)
![Chef Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/15-chef.png)
![Finance Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/16-finance.png)
![Apparel Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/17-apparel.png)
![Engineering Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/18-engineering.png)
![Accounting Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/19-accounting.png)
![Construction Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/20-construction.png)
![Public Relations Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/21-pr.png)
![Banking Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/22-banking.png)
![Arts Word Cloud](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/wordclouds/23-arts.png)

## Methods ##

### [Wordcloud](https://amueller.github.io/word_cloud/) ###
WordClouds are a visually pleasing way to get a high-level overview of themes or topics from bodies of text. WordCloud counts the frequency of words and can be used to generate the images above. The larger the word the more frequently it shows up inside the provided text. It is used in conjunction with STOPWORDS which filter out common words that do not have any particular meaning or value such as the, and, a, by, use, etc. In our code, we also added some additional words that were showing up inside our WordClouds such as city, state, name, and hr.

### [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) ###
Beautiful Soup is a Python library that is commonly used for web scraping and parsing HTML or XML documents. For our research, Beautiful Soup was used to extract job titles from HTML content, specifically the 'Job Description' column in the provided dataset. The use of Beautiful Soup for extracting job titles is relevant for understanding potential factors that could have influenced the accuracy of the model. Factors influencing model accuracy include overlapping subjects in categories like Information-Technology and Teacher, shared terminology among categories, decreased accuracy when using frequent words as features, and the need for a larger and balanced dataset. Job titles alone proved to be a better predictor.

### [TfidfVectorizer - Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer) ###
We used our derived Job Titles column to populate a vocabulary dictionary and generate features using this vectorizer. We then used this vectorizer to transform our X_train and X_test data.

### [Multinomial Naive Bayes Classifier - Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) ###
The Multinomial Naive Bayes classifier provides a method for classifying text by word counts. After using TfidfVectorizer to generate a vectorized version of X_train and X_test, we fit and tested our model with this vectorized data.

## Results ##
We were able to use a resume’s Job Title as a predictor for its Job Category with an accuracy score of about 62%.

**Heat Map of Confusion Matrix**
![Heat Map of confusion matrix](https://github.com/emmariegeo/cst383final/blob/33eb3cf62abfd012734aefa0b5a75a2f5cf7e416/report_images/heatmap.png)

## Discussion ##

The categories where we saw the least accuracy were Information-Technology, Teacher, Advocate, Agriculture, and Chef. The inaccuracy in the Teacher category could be caused by the wide range of subjects a teacher might specialize in being picked up by other categories. Looking at the generated word clouds of frequent words, there is a large amount of overlap in the Information-Technology, Advocate, Consultant, and Engineering categories. Although these frequencies were not used as a predictor, they reflect that the shared terminology across these categories could have impacted the model’s accuracy. The terms most frequently used in the Agriculture category did not appear to be highly specific to the category itself and seemed more general to resume terminology. This might stem from having a smaller number of Agriculture category samples within the Resume dataset.
 
When we added the Frequent Words generated by WORDCLOUD as features, we found that the accuracy decreased. This could be because the frequency of words is already being analyzed through our model comparison. It could also be caused by a high overlap of frequently used words in the resumes of different categories. It seems that Job Title alone was a better predictor.

Increasing the dataset size and training set size could improve the accuracy of our model. When selecting a random subset of our resume samples, we might create a training set with very few entries in one category. If we were to both increase our sample size and balance out the number of resumes per category, we could likely achieve a more accurate result. 

## Summary ##

The Multinomial Naive Bayes classifier allowed us to predict the Job Category associated with a resume by using the resume’s primary job title as a predictor. The job titles proved to be a key factor in predicting job categories from resumes, providing valuable insights for the model. However, we encountered variations in accuracy across different job categories. The Information-Technology, Teacher, Advocate, Agriculture, and Chef categories exhibited lower accuracy, possibly due to overlapping subjects in teaching and shared terminology among other fields. Our analysis of frequent words revealed interesting patterns, showing significant overlap in terminology among Information-Technology, Advocate, Consultant, and Engineering categories. Despite these challenges, job titles remained a strong predictor overall. To enhance our model's accuracy, we can focus on refining the job title extraction process and addressing the nuances of shared terminology. By doing so, we can further improve our ability to predict job categories accurately from resumes.

## References ## 

Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.  

Richardson, L. (2023). Beautiful soup. Beautiful Soup: We called him Tortoise because he taught us. https://www.crummy.com/software/BeautifulSoup/  

WordCloud for python documentation. WordCloud for Python documentation - wordcloud 1.8.1 documentation. (2020). https://amueller.github.io/word_cloud/  
