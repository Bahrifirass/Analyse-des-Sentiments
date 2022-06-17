# importing packages
import streamlit as st
import plotly.io as pio
pio.renderers.default='browser'
import numpy as np
import pandas as pd
import seaborn as sns
#import plotly as px
import plotly.express as px
import matplotlib.pyplot as plt
#%matplotlib inline
import re
from bs4 import BeautifulSoup as bs
import requests
import string
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords,wordnet 
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
#pyLDAvis.enable_notebook()
import warnings
warnings.filterwarnings("ignore")
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

def scrape_reviews(hotel_linkname,total_pages ):
     #Create empty lists to put in reviewers' information as well as all of the positive & negative reviews 
    info = []
    positive = []
    negative = []
     
     #bookings.com reviews link
    url = "https://www.booking.com/reviews/fr/hotel/"+hotel_linkname+".html?r_lang=all&page=" 
    page_number = 1
    #Use a while loop to scrape all the pages 
    print('connecting to'+url)
    while page_number <= total_pages:
        print(page_number)
        
        page = requests.get(url + str(page_number)) #retrieve data from server
        soup = bs(page.text, "html.parser") # initiate a beautifulsoup object using the html source and Python's html.parser
        review_box = soup.find('ul',{'class':'review_list'}) 
        #ratings
        ratings = [i.text.strip() for i in review_box.find_all('span',{'class':'review-score-badge'})]
         
         #reviewer_info
        print('reviews')
        reviewer_info = [i.text.strip() for i in review_box.find_all('span',{'itemprop':'name'})]
        reviewer_name = reviewer_info[0::3]
        reviewer_country = reviewer_info[1::3]
        general_review = reviewer_info[2::3]
        # reviewer_review_times
        review_times = [i.text.strip() for i in review_box.find_all('div',{'class':'review_item_user_review_count'})]
        # review_date
        review_date = [i.text.strip().strip('Reviewed: ') for i in review_box.find_all('p',{'class':'review_item_date'})]
        # reviewer_tag
        reviewer_tag = [i.text.strip().replace('\n\n\n','').replace('•',',').lstrip(', ') for i in review_box.find_all('ul',{'class':'review_item_info_tags'})]
        # positive_review
        positive_review = [i.text.strip('눇').strip() for i in review_box.find_all('p',{'class':'review_pos'})]
        # negative_review
        negative_review = [i.text.strip('눉').strip() for i in review_box.find_all('p',{'class':'review_neg'})]
        # append all reviewers' info into one list
        print('append')
        for i in range(len(reviewer_name)):
            info.append([ratings[i],reviewer_name[i],reviewer_country[i],general_review[i], 
            review_times[i],review_date[i],reviewer_tag[i]])
        # build positive review list
        for i in range(len(positive_review)):
            positive.append(positive_review[i])
        # build negative review list
        for i in range(len(negative_review)):
            negative.append(negative_review[i])
        # page change
        page_number +=1
    #Reviewer_info df
    print('faming')
    reviewer_info = pd.DataFrame(info,
    columns = ['Rating','Name','Country','Overall_review','Review_times','Review_date','Review_tags'])
    reviewer_info['Rating'] = pd.to_numeric(reviewer_info['Rating'] )
    reviewer_info['Review_times'] = pd.to_numeric(reviewer_info['Review_times'].apply(lambda x:re.findall("\d+", x)[0]))
    reviewer_info['Review_date'] = pd.to_datetime(reviewer_info['Review_date'])
     
     #positive & negative reviews dfs
    pos_reviews = pd.DataFrame(positive,columns = ['positive_reviews'])
    neg_reviews = pd.DataFrame(negative,columns = ['negative_reviews'])
     
    return reviewer_info, pos_reviews, neg_reviews



def show_data(df):
    print("The length of the dataframe is: {}".format(len(df)))
    print("Total NAs: {}".format(reviewer_info.isnull().sum().sum()))
    return df.head()


reviewer_info, pos_reviews, neg_reviews = scrape_reviews('westside-arc-de-triomphe',total_pages = 3 )

show_data(reviewer_info) #reviewers’ basic information
show_data(pos_reviews)   #Positive reviews
show_data(neg_reviews)   #Negative reviews


#https://www.booking.com/reviews/in/hotel/ramada-caravela-beach-resort.en-gb.html?page=1





# DATA ANALYSIS
   # Pos vs Neg

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
values = [len(pos_reviews), len(neg_reviews)]
ax.pie(values, labels = ['Number of Positive Reviews', 'Number of Negative Reviews'],colors=['gold', 'lightcoral'],
 shadow=True,
 startangle=90, 
 autopct='%1.2f%%')
ax.axis('equal')
plt.title('Positive Reviews Vs. Negative Reviews');


     # Top 10 reviewers country of origin
 
top10_list = reviewer_info['Country'].tolist() 
top10 = reviewer_info[reviewer_info.Country.isin(top10_list)]
fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
ax = sns.violinplot(x = 'Country', y = 'Rating',data = top10, order = top10_list,linewidth = 2) 
plt.suptitle('Distribution of Ratings by Country') 
plt.xticks(rotation=90);



     #Distribution of review tags count for each trip type
 
#Define tag list
tag_list = ['Business','Leisure','Group','Couple','Family','friends','Solo']
#Count for each review tag
tag_counts = []
for tag in tag_list:
 counts = reviewer_info['Review_tags'].str.count(tag).sum()
 tag_counts.append(counts)
#Convert to a dataframe
trip_type = pd.DataFrame({'Trip Type':tag_list,'Counts':tag_counts}).sort_values('Counts',ascending = False)
#Visualize the trip type counts from Review_tags
fig = px.bar(trip_type, x='Trip Type', y='Counts', title='Review Tags Counts for each Trip Type')
fig.show()


# wordnet and treebank have different tagging systems
# Create a function to define a mapping between wordnet tags and POS tags 
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
     return wordnet.ADJ
    elif pos_tag.startswith('V'):
     return wordnet.VERB
    elif pos_tag.startswith('N'):
     return wordnet.NOUN
    elif pos_tag.startswith('R'):
     return wordnet.ADV
 
    else:
     return wordnet.NOUN # default, return wordnet tag “NOUN”
#Create a function to lemmatize tokens in the reviews
def lemmatized_tokens(text):
 text = text.lower()
 pattern = r'\b[a-zA-Z]{3,}\b' 
 tokens = nltk.regexp_tokenize(text, pattern) # tokenize the text
 tagged_tokens = nltk.pos_tag(tokens) # a list of tuples (word, pos_tag)
 
 stop_words = stopwords.words('english')
 new_stopwords = ['hotel','everything','anything','nothing','thing','need',
 'good','great','excellent','perfect','much','even','really'] #customize extra stop_words
 stop_words.extend(new_stopwords)
 stop_words = set(stop_words)
 
 wordnet_lemmatizer = WordNetLemmatizer()
 # get lemmatized tokens #call function “get_wordnet_pos”
 lemmatized_words=[wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
 # tagged_tokens is a list of tuples (word, tag)
 for (word, tag) in tagged_tokens \
 # remove stop words
     if word not in stop_words and \
     # remove punctuations
     word not in string.punctuation]
 return lemmatized_words


#Create a function to generate wordcloud
def wordcloud(review_df, review_colname, color, title):
     ''' 
     INPUTS:
     reivew_df — dataframe, positive or negative reviews
     review_colname — column name, positive or negative review
     color — background color of worldcloud
     title — title of the wordcloud
     OUTPUT:
     Wordcloud visuazliation
     '''

     text = review_df[review_colname].tolist()
     text_str = ' '.join(lemmatized_tokens(' '.join(text))) #call function "lemmatized_tokens"
     wordcloud = WordCloud(collocations = False, background_color = color, width=1600, height=800, margin=2,min_font_size=20).generate(text_str)
     plt.figure(figsize = (15, 10))
     plt.imshow(wordcloud, interpolation = 'bilinear')
     plt.axis("off")
     plt.figtext(.5,.8,title,fontsize = 20, ha='center')
     plt.show() 
     
         
        # Wordcoulds for Positive Reviews
wordcloud(pos_reviews,'positive_reviews', 'white','Positive Reviews: ')
                # # WordCoulds for Negative Reviews
wordcloud(neg_reviews,'negative_reviews', 'black', 'Negative Reviews:')
  
  
  
   #Create a function to get the subjectivity
def subjectivity(text): 
 return TextBlob(text).sentiment.subjectivity
#Create a function to get the polarity
def polarity(text): 
 return TextBlob(text).sentiment.polarity
#Create two new columns
reviewer_info['Subjectivity'] = reviewer_info['Overall_review'].apply(subjectivity)
reviewer_info['Polarity'] = reviewer_info['Overall_review'].apply(polarity)
#################################################################################
#Create a function to compute the negative, neutral and positive analysis
def getAnalysis(score):
 if score <0:
    return 'Negative'
 elif score == 0:
  return 'Neutral'
 else:
  return 'Positive'
reviewer_info['Analysis'] = reviewer_info['Polarity'].apply(getAnalysis)
#################################################################################
# plot the polarity and subjectivity
fig = px.scatter(reviewer_info, x='Polarity', y='Subjectivity', color = 'Analysis',size='Subjectivity')
#add a vertical line at x=0 for Netural Reviews
#fig.update_layout(title='Sentiment Analysis',shapes=[dict(type= 'line',yref= 'paper', y0= 0, y1= 1, xref= 'x', x0= 0, x1= 0)])
fig.show()



#Create a function to build the optimal LDA model
def optimal_lda_model(df_review, review_colname):
     '''
     INPUTS:
     df_review — dataframe that contains the reviews
     review_colname: name of column that contains reviews
     
     OUTPUTS:
     lda_tfidf — Latent Dirichlet Allocation (LDA) model
     dtm_tfidf — document-term matrix in the tfidf format
     tfidf_vectorizer — word frequency in the reviews
     A graph comparing LDA Model Performance Scores with different params
     '''
     docs_raw = df_review[review_colname].tolist()
    #************ Step 1: Convert to document-term matrix ************#
    #Transform text to vector form using the vectorizer object 
     tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                    stop_words = 'english',
                                    lowercase = True,
                                    token_pattern = r'\b[a-zA-Z]{3,}\b', # num chars > 3 to avoid some meaningless words
                                    max_df = 0.9, # discard words that appear in > 90% of the reviews
                                    min_df = 3) # discard words that appear in < 10 reviews
    #apply transformation
     tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())
    #convert to document-term matrix
     dtm_tfidf = tfidf_vectorizer.fit_transform(docs_raw)
     print("The shape of the tfidf is {}, meaning that there are {} {} and {} tokens made through the filtering process.".\
    format(dtm_tfidf.shape,dtm_tfidf.shape[0], review_colname, dtm_tfidf.shape[1]))
    #******* Step 2: GridSearch & parameter tuning to find the optimal LDA model *******#
    # Define Search Param
     search_params = {'n_components': [5, 10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
    # Init the Model
     lda = LatentDirichletAllocation()
    # Init Grid Search Class
     model = GridSearchCV(lda, param_grid=search_params)
    # Do the Grid Search
     model.fit(dtm_tfidf)
    #***** Step 3: Output the optimal lda model and its parameters *****#
    # Best Model
     best_lda_model = model.best_estimator_
    # Model Parameters
     print("Best Model’s Params: ", model.best_params_)
    # Log Likelihood Score: Higher the better
     print("Model Log Likelihood Score: ", model.best_score_)
    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
     print("Model Perplexity: ", best_lda_model.perplexity(dtm_tfidf))
    #*********** Step 4: Compare LDA Model Performance Scores ***********#
    #Get Log Likelyhoods from Grid Search Output
     gscore=model.fit(dtm_tfidf).cv_results_
     n_topics = [5, 10, 15, 20, 25, 30]
     log_likelyhoods_5 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.5]
     log_likelyhoods_7 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.7]
     log_likelyhoods_9 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.9]
    # Show graph
     plt.figure(figsize=(12, 8))
     plt.plot(n_topics, log_likelyhoods_5, label='0.5')
     plt.plot(n_topics, log_likelyhoods_7, label='0.7')
     plt.plot(n_topics, log_likelyhoods_9, label='0.9')
     plt.title("Choosing Optimal LDA Model")
     plt.xlabel("Num Topics")
     plt.ylabel("Log Likelyhood Scores")
     plt.legend(title='Learning decay', loc='best')
     plt.show()
 
     return best_lda_model, dtm_tfidf, tfidf_vectorizer

best_lda_model, dtm_tfidf, tfidf_vectorizer = optimal_lda_model(neg_reviews, 'negative_reviews')

################################################################

#Create a function to inspect the topics we created 
def display_topics(model, feature_names, n_top_words):
     '''
     INPUTS:
     model — the model we created
     feature_names — tells us what word each column in the matric represents
     n_top_words — number of top words to display
     OUTPUTS:
     a dataframe that contains the topics we created and the weights of each token
     '''
     topic_dict = {}
     for topic_idx, topic in enumerate(model.components_):
      #   print(topic_idx, topic)
         topic_dict["Topic %d words" % (topic_idx+1)]= ['{}'.format(feature_names[i])
                                                         for i in topic.argsort()[:-n_top_words - 1:-1]]
         topic_dict["Topic %d weights" % (topic_idx+1)]= ['{:.1f}'.format(topic[i])
                                                        for i in topic.argsort()[:-n_top_words - 1:-1]]
     return pd.DataFrame(topic_dict)
display_topics(best_lda_model, tfidf_vectorizer.get_feature_names(), n_top_words = 20)


# Topic Modelling Visualization for the Negative Reviews

vis_data=pyLDAvis.sklearn.prepare(best_lda_model, dtm_tfidf, tfidf_vectorizer)
pyLDAvis.save_html(vis_data, 'lda.html')

st.title('Sentiment Analysis')


