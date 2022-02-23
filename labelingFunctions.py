import os
from sklearn.model_selection import train_test_split
import pandas as pd
#import snorkel
from snorkel.labeling import labeling_function
from textblob import TextBlob
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
import matplotlib.pyplot as plt
from snorkel.labeling import LFAnalysis
from snorkel.preprocess import preprocessor
from snorkel.classification.data import DictDataset, DictDataLoader

ABSTAIN = -1
DISORDER = 1
NONDISORDER = 0

EDkeywords = ["binge", "binging", "purge", "purging", "fat", "anorexic", "ana", "mia", "bulimia", "trigger warning", "tw", "compulsive", "can't stop", "please help", "body check", "starving", "feel so fat", "hate my body", "wish i was skinny", "want to be skinny", "thinspo"]

nonEDkeywords = ["in recovery", "getting help", "am eating intuitively"]

filename = "total_labeled_data"
csv = pd.read_csv(filename)
text = csv.post.tolist()
newtext = []
newlabels = []
labels = csv.labels.tolist()
for i in range(len(text)):
    if str(text[i]).lower() != "nan":
        newtext.append(str(text[i]))
        newlabels.append(int(labels[i]))
        
df = pd.DataFrame({"labels": newlabels, "text": newtext});
df_train,df_test = train_test_split(df,train_size=0.9)



#print(df_train["text"].sample(20, random_state=2))

@labeling_function()
def contains_EDkeywords(x):
	for keyword in EDkeywords: 
		if keyword in x.text.lower():
			return DISORDER

@labeling_function()
def contains_nonEDkeywords(x):
    for keyword in nonEDkeywords:
        if keyword in x.text.lower():
            return NONDISORDER

#USE THIS FOR ANOTHER TEST: SEMI-ARBITRARY LABELS
#@labeling_function()
#def prelabel(x):
#	if x.label == 1:
#		return DISORDER
#	else:
#		return NONDISORDER
  
@labeling_function()
def lf_textblob_polarity(x):
    return NONDISORDER if TextBlob(x.text.lower()).sentiment.polarity > 0.3 else DISORDER
    

lfs = [contains_EDkeywords, contains_nonEDkeywords, lf_textblob_polarity]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)

#label_model = LabelModel(cardinality=2, verbose=True)
#label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
#df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")
#df_train = df_train[df_train.label != ABSTAIN]

LFAnalysis(L=L_train, lfs=lfs).lf_summary()

def plot_label_frequency(L):
    plt.hist((L != ABSTAIN).sum(axis=1), density=True, bins=range(L.shape[1]))
    plt.xlabel("Number of labels")
    plt.ylabel("Fraction of dataset")
    plt.show()
    
plot_label_frequency(L_train)

