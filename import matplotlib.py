import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import seaborn as sns
import PyPDF2


%matplotlib inline


data_loc = "C:/Users/pauli/OneDrive/Área de Trabalho/git_repo/ai_nlp/data/"

import os 
files = os.listdir(data_loc)

pdf_reader = PyPDF2.PdfFileReader(data_loc+files[0])


pdf_reader.documentInfo

print("Paula")



vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform()

