# Twitter sentiment analysis with BERT

This repo contains resources to further pre-train and fine-tune the BERT model for twitter sentiment analysis.
The code was written/adapted prior to tensorflow team lauching the "classify text with bert" [guide](https://www.tensorflow.org/tutorials/text/classify_text_with_bert). 
If you'd like a simpler method for training the BERT model for TSA, I'd recommend you to follow the mentioned guide. 
I plan to implement this code according to such, in the near future.

## BERT vs other approaches
Model       | Avg Recall  |  *F1* | Accuracy
 :---       |     :---:   | :---: |  :---:
BB_twtr     |    0.681    | 0.685 |  0.658
DataStories |    0.681    | 0.677 |  0.651
VADER       |    0.524    | 0.567 |  0.530
BERT_M4     |    0.711    | 0.703 |  0.697
