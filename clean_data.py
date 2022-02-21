import pandas as pd
import os
import re
import emoji


def findURLS(string):
  
    # findall() has been used 
    # with valid conditions for urls in string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)      
    return [x[0] for x in url]


def extract_emojis(s):
  return ''.join(c for c in s if c in emoji.UNICODE_EMOJI['en'])

labels = []
data = []
directory = 'ed_subreddit_data_negative'

for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	if os.path.isfile(f):
		csv = pd.read_csv(f)
		bodies = csv.body.tolist()
		i = 0
		for body in bodies:
			i += 1
			print(i)
			body = str(body)
			body.replace("\n", "")
			body.replace("&#x200B;", "")
			urls = findURLS(body)
			for url in urls: 
				body.replace(url, "")
			emojis = extract_emojis(body)
			for e in emojis: 
				body.replace(e, "")
			data.append(body)
			labels.append(0)
df = pd.DataFrame({"labels": labels, "post": data})
df.to_csv("negative_labeled_data")


		



