import pandas as pd
import os
import re
import emoji


def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def strip_emoji(text):
    new_text = re.sub(emoji.get_emoji_regexp(), r"", text)
    return new_text

labels = []
data = []
directories = ['ed_subreddit_data_positive', 'ed_subreddit_data_negative']


file = 0
for directory in directories: 
	for filename in os.listdir(directory):
		print("file: ", filename)
		file += 1
		f = os.path.join(directory, filename)
		if os.path.isfile(f):
			try: 
				csv = pd.read_csv(f)
			except:
				continue
			bodies = csv.body.tolist()
			post = 0
			print("total num posts: ",len(bodies))
			for body in bodies:
				print("postno: ", post)
				print(body)
				post += 1
				body = str(body)
				body.replace("\n", "")
				body.replace("&#x200B;", "")
				body = remove_URL(body)
				body = remove_emoji(body)
				body = strip_emoji(body)
				data.append(body)
				if directory == 'ed_subreddit_data_positive':
					labels.append(1)
				else: 
					labels.append(0)
print("ALL DONE")
df = pd.DataFrame({"labels": labels, "post": data})
print(df)
df.to_csv("total_labeled_data")



		



