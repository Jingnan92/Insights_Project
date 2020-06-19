## Market Value Estimator: Empowering the Job Search and Salary Negotiation

### Insight SEA DS 2020B Project

When candidates go to job market, they typically have two choices: job boards and company official career sites. In both cases, information about the estimated salary is not always availabe! Job boards, like Glassdoor or LinkedIn, rely on users' reported salary to provide estimations on part of their posted job openings. But if no one reported their salary information on certain type of job in certain company, estimated salary is not provided. Altrnatively, salary is not listed on company offical career sites in most cases. Therefore, this project aims to provide an estimate on expected salary using information that is always available to job seekers, empowering their job-seeking activities and salay negotiation capabilities.

<li> 1. Data </li>

Dataset for this project includes 1,000+ data scientist related job postings from Glassdoor. It includes information about the hiring comany (e.g. location, industry, size, etc.) and about the hiring positions (e.g. job titile, job descriptions, estimated salary, etc.). For more information about the data, please see [Data](https://github.com/Jingnan92/Insights_Project/blob/master/Data), and for data cleaning process, please see the Python [script](https://github.com/Jingnan92/Insights_Project/blob/master/Scripts/Scripts_Part01.py).



<li> 2. Model </li>

NLP technique is used to proprocess the data. A word2vec model is trained with all the words included in this dataset to get word embeddings. TF-IDF weighted word2vec aggreagates the word embeddings to document embeddings. Then data is splitted into train and test set and fitted into two Random Forest Models. These two models predict the maximum and minimum expected salary seperately. Model accuray is selected to evaluate the perfoman performance, and it shows that the accuracy for Random Forest models for maximum and minimum salary prediction is 78.70% and 72.46% respectively (see more details from the Python [script](https://github.com/Jingnan92/Insights_Project/blob/master/Scripts/Scripts_Part02.py).



<li> 3. Web App </li>

The web app is built with Streamlit and deployed on Amazon Web Services(AWS). It allows users to directly copy and past information from job postings and calculate their expected salary range. Check this [Python file](https://github.com/Jingnan92/Insights_Project/blob/master/app.py) for more detail. 
