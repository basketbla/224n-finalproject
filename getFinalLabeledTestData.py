import pandas as pd
import os
import re
import emoji




dfchristine = pd.read_csv("/Users/christinemanegan/Desktop/classes/CS224N/224n-finalproject/labeled_christine.csv")
dfrhett = pd.read_csv("/Users/christinemanegan/Desktop/classes/CS224N/224n-finalproject/labeled_rhett.csv")

finalLabeledTestData = pd.concat([dfrhett, dfchristine])
rows = finalLabeledTestData.ogindex.tolist()

dftotal_labeled_nonan = pd.read_csv("/Users/christinemanegan/Desktop/classes/CS224N/224n-finalproject/total_labeled_nonan.csv")
dftotal_labeled_nonan.drop(rows, inplace=True)
dftotal_labeled_nonan.reset_index(drop=True, inplace=True)
finalLabeledTestData.reset_index(inplace=True)

finalLabeledTestData.to_csv("FINAL_labeled_test_data.csv")
dftotal_labeled_nonan.to_csv("FINAL_train_data.csv")