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

filename = "total_labeled_nonan.csv"
csv = pd.read_csv(filename)
text = csv.text.tolist()
labels = csv.label.tolist()
newlabels = []
newtext = []
for i in range(len(text)):
    newtext.append(str(text[i]))
    newlabels.append(int(labels[i]))
        
df = pd.DataFrame({"labels": newlabels, "text": newtext});
df_train,df_test = train_test_split(df,train_size=0.9)



#print(df_train["text"].sample(20, random_state=2))

@labeling_function()
def contains_EDkeywords(x):
    string = str(x.text).lower()
    for keyword in EDkeywords: 
        if keyword in string:
            return DISORDER
    return ABSTAIN

@labeling_function()
def contains_nonEDkeywords(x):
    string = str(x.text).lower()
    for keyword in nonEDkeywords:
        print(x.text.lower())
        if keyword in string:
            return NONDISORDER
    return ABSTAIN

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


print(L_train)

ed_keywords, noned_keywords, texblob_polarity = (L_train != ABSTAIN).mean(axis=0)
# print(f"ed_keywords coverage: {ed_keywords * 100:.1f}%")
# print(f"noned_keywords coverage: {noned_keywords * 100:.1f}%")
# print(f"textblob_polarity coverage: {texblob_polarity * 100:.1f}%")

from snorkel.labeling import LFAnalysis

analysis_df = LFAnalysis(L=L_train, lfs=lfs).lf_summary()

# analysis_df.to_csv(r'/Users/christinemanegan/Desktop/classes/CS224N/224n-finalproject/firstPassLearningFuncsAnalysis.csv', index = False, header=True)

from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

from snorkel.labeling import filter_unlabeled_dataframe

probs_train = label_model.predict_proba(L=L_train)

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)
df_train_filtered.to_csv(r'/Users/christinemanegan/Desktop/classes/CS224N/224n-finalproject/firstPassSnorkelLabels.csv', index = False, header=True)

# LFAnalysis(L=L_train, lfs=lfs).lf_summary()

# def plot_label_frequency(L):
#     plt.hist((L != ABSTAIN).sum(axis=1), density=True, bins=range(L.shape[1]))
#     plt.xlabel("Number of labels")
#     plt.ylabel("Fraction of dataset")
#     plt.show()
    
# plot_label_frequency(L_train)

