import pandas as pd
import os
import re
import emoji

df = pd.read_csv("/Users/christinemanegan/Desktop/classes/CS224N/224n-finalproject/total_labeled_nonan.csv")
posdf = df.loc[df['label'] == 1]
negdf = df.loc[df['label'] == 0]

posdf = posdf.sample(n=100)
negdf= negdf.sample(n=100)

result = pd.concat([posdf, negdf])
result.to_csv("labeled_test_data.csv")
print(result)


