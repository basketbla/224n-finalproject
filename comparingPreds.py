import pandas as pd

df_2 = pd.read_csv("bert_base_snorkel_2_analysis.csv")
df_1 = pd.read_csv("bert_base_snorkel_4_analysis.csv")

seenTexts = set()
issues = []
texts = []

fp_overlap = 0
fn_overlap = 0
for t in df_1.text.to_list():
	seenTexts.add(t)
df_2_text = df_2.text.to_list()
for i in range(len(df_2_text)):
	if df_2_text[i] not in seenTexts:
		issues.append(df_2.issue.to_list()[i])
		texts.append(df_2_text[i])
	else:
		if (df_2.issue.to_list()[i] == "fp"):
			fp_overlap += 1
		else:
			fn_overlap += 1
fp_percent = fp_overlap/len(df_2_text)
fn_percent = fn_overlap/len(df_2_text)
print(fp_percent)
print(fn_percent)
#diffMapdf = pd.DataFrame({"issue": issues, "text": texts})
#diffMapdf.to_csv('bbs4_vs_bbs3.csv')

