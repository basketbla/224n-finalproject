import pandas as pd
import os
import re
import emoji


def findURLS(string):
  
    # findall() has been used 
    # with valid conditions for urls in string
    words = []
    for word in string: 
    	if "http" in word or "www" in word: 
    		words.append(word)
    #regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    #url = re.findall(regex,string)      
    #return [x[0] for x in url]
    return words


def extract_emojis(s):
  return ''.join(c for c in s if c in emoji.UNICODE_EMOJI['en'])

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
				urls = findURLS(body)
				for url in urls: 
					body.replace(url, "")
				emojis = extract_emojis(body)
				for e in emojis: 
					body.replace(e, "")
				data.append(body)
				if directory == 'ed_subreddit_data_positive':
					labels.append(1)
				else: 
					labels.append(0)
print("ALL DONE")
df = pd.DataFrame({"labels": labels, "post": data})
print(df)
df.to_csv("total_labeled_data")



		



