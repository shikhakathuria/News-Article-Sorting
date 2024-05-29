import streamlit as st
import pickle
import re
from bs4 import BeautifulSoup
import string
import wordninja as wn
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import nltk 
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.tokenize import RegexpTokenizer



pickle_model = open("cnb.pk", "rb")
model = pickle.load(pickle_model)

pickel_tfidf_vectorizer = open("TfIdf_Vectorizer.pk","rb")
tfidf_vectorizer = pickle.load(pickel_tfidf_vectorizer)



def predict_news(sentence):
    # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # TfIdf_Vectorizer = TfidfVectorizer(tokenizer = token.tokenize)

        # make smallercase
    sentence = sentence.lower()
        
    # remove emails
    sentence = re.sub(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", '', sentence)

    # remove mentions
    sentence = re.sub(r"@[A-Za-z0-9]+","", sentence)
    
    # Remove html
    sentence = BeautifulSoup(sentence, 'lxml').get_text().strip()
    
    # Remove URL
    sentence = re.sub(r'https?://\S+|www\.\S+', '', sentence)
        
    # Removing punctutation, string.punctuation in python consists of !"#$%&\'()*+,-./:;<=>?@[\\]^_
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    
    # Remove non-alphabetic characters
    sentence = re.sub(r'[^a-zA-Z ]', '', sentence)

    # Spliting words into two meaningful word
    sentence = ' '.join(wn.split(str(sentence)))

    # Correct the sentence
    sentence =  str(TextBlob(sentence).correct())
    
    # decontracted
    
    ## specific
    sentence = re.sub(r"wont", "will not", sentence)
    sentence = re.sub(r"wouldnt", "would not", sentence)
    sentence = re.sub(r"shouldnt", "should not", sentence)
    sentence = re.sub(r"couldnt", "could not", sentence)
    sentence = re.sub(r"cudnt", "could not", sentence)
    sentence = re.sub(r"cant", "can not", sentence)
    sentence = re.sub(r"dont", "do not", sentence)
    sentence = re.sub(r"doesnt", "does not", sentence)
    sentence = re.sub(r"didnt", "did not", sentence)
    sentence = re.sub(r"wasnt", "was not", sentence)
    sentence = re.sub(r"werent", "were not", sentence)
    sentence = re.sub(r"havent", "have not", sentence)
    sentence = re.sub(r"hadnt", "had not", sentence)

    ## general
    sentence = re.sub(r"n\ t", " not", sentence)
    sentence = re.sub(r"\re", " are", sentence)
    sentence = re.sub(r"\ s ", " is ", sentence) 
    sentence = re.sub(r"\ d ", " would ", sentence)
    sentence = re.sub(r"\ ll ", " will ", sentence)
    sentence = re.sub(r"\dunno", "do not ", sentence)
    sentence = re.sub(r"ive ", "i have ", sentence)
    sentence = re.sub(r"im ", "i am ", sentence)
    sentence = re.sub(r"i m ", "i am ", sentence)
    sentence = re.sub(r" w ", " with ", sentence)


    # Lemmatize
    lm = WordNetLemmatizer()
    snt = ' '.join([lm.lemmatize(i) for i in sentence.split()])
    
    return model.predict(tfidf_vectorizer.transform([snt]))[0]


def main():
    st.title("News Article Classifier")
    html_template = """
    <div style="background-color:tomato;padding=10px" >
    <h2 stype="color:white;text-align:center"> News Arctcle Sorting App
    </h2>
    </div>
    """
    # st.markdown(html_template,unsafe_allow_html=True)
    news_artcile = st.text_area("New Arcticle","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_news(news_artcile)
    st.success('The article is {}'.format(result))

if __name__=='__main__':
    main()



    