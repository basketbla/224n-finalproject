import praw
import pandas as pd


#authenticate
reddit = praw.Reddit(client_id='uOU7fhX5xBkf2ue5U7lZGw', client_secret='cygPTdRPAHemzNovgjnXFqeWt87iUQ', user_agent='eating disorder data for ML research study')
posts = []

#get posts from subreddit and load into csv
hot_posts = reddit.subreddit('Eatingdisordersover30').hot(limit=10000)
for post in hot_posts:
	 posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
print(posts)
posts.to_csv(r'/Users/christinemanegan/Desktop/classes/CS224N/Eatingdisordersover30.csv', index = False, header=True)
