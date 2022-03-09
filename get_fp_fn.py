

import pandas as pd 


true_result = pd.read_csv("FINAL_labeled_test_data.csv")
true_result_labels = true_result.label.to_list();
true_result_text = true_result.text.to_list();

test_result = pd.read_csv("bert_base_snorkel_4_preds.csv")
test_result_labels = test_result.label.to_list();

fp = 0
fn = 0
issue = []
text = []

for i in range(len(true_result_labels)):
	if true_result_labels[i] > test_result_labels[i]:
		fn += 1
		text.append(true_result_text[i])
		issue.append("fn")
	if true_result_labels[i] < test_result_labels[i]:
		fp += 1
		text.append(true_result_text[i])
		issue.append("fp")
df = pd.DataFrame({"fp/fn": issue, "text": text})
df.to_csv("bert_base_snorkel_4_analysis.csv")