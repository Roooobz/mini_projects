# NB Model

Naïve Bayes model trained on the tagged_selections_by_sentence.csv to identify the presence of Greetings in highlighted texts from the original texts. 
First, the Selected text is preprocessed with TF-IDF. Then multinomial Naïve Bayes is applied to the results.
Training and test data are randomly chosen at a 70/30 split.

Results are displayed as an accuracy precentage.
