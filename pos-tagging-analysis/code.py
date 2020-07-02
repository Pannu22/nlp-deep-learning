# --------------
import pandas as pd
import nltk

df = pd.read_csv(path)

def pos_tagger(sent):
    sent = sent.split()
    sent_tag = nltk.pos_tag(sent)
    return sent_tag

tagged_titles = df['nominee'].apply(lambda emmy_nominee: pos_tagger(emmy_nominee))
tagged_titles_df = pd.DataFrame(data = tagged_titles)
print(tagged_titles_df.head())


# --------------
import matplotlib.pyplot as plt

# Function to create tags

def count_tags(title_with_tags):
    tag_count = {}
    for word, tag in title_with_tags:
        if tag in tag_count:
            tag_count[tag] += 1
        else:
            tag_count[tag] = 1
    return(tag_count)


# Mapping the counts with the tags
tagged_titles_df['tag_counts'] = tagged_titles_df['nominee'].map(count_tags)


# Tagset containing all the possible tags
tag_set = list(set([tag for tags in tagged_titles_df['tag_counts'] for tag in tags]))

# Creating tag column frequency for each tags
for tag in tag_set:
    tagged_titles_df[tag] = tagged_titles_df['tag_counts'].map(lambda x: x.get(tag, 0))


# Subsetting the dataframe to contain only the tagset columns    
top_pos=tagged_titles_df[tag_set]

# Sorting and storing the top 10 frequent tags
top_pos=top_pos.sum().sort_values().tail(10)
    
    
# Plotting the barplot of the tag frequency   
title = 'Frequency of POS Tags in Show Titles'    
top_pos.plot(kind='barh', figsize=(18,10), title=title)


plt.show()


# --------------
# Function to create vocabulary of the tags
def vocab_creator(tagged_titles):
    vocab = {}

    for row in tagged_titles['nominee']:
        for word, tag in row:
            if word in vocab:
                if tag in vocab[word]:
                    vocab[word][tag] += 1
                else:
                    vocab[word][tag] = 1
            else:
                vocab[word] = {tag: 1}
            
    return vocab    
            
# Creating vocab of our tagged titles dataframe
vocab= vocab_creator(tagged_titles_df)

vocab_df = pd.DataFrame.from_dict(vocab, orient = 'Index')
vocab_df.fillna(value = 0, inplace = True)
top_verb_nominee = vocab_df['VBG'].sort_values().tail(10)

title = 'Top Verbs in Show Titles'    
top_verb_nominee.plot(kind='barh', figsize=(18,10), title=title)
plt.show()

top_noun_nominee = vocab_df['NN'].sort_values().tail(10)

title = 'Top Nouns in Show Titles'    
top_noun_nominee.plot(kind='barh', figsize=(18,10), title=title)
plt.show()


# --------------
# Subsetting comedy winners
new_df=df[(df['winner']==1) & (df['category'].str.contains('Comedy'))]

# Mapping the position tags of the winners
tagged_titles_winner = new_df['nominee'].str.split().map(pos_tag)

# Creating a dataframe
tagged_titles_winner_df=pd.DataFrame(tagged_titles_winner)

# Creating a vocabulary of the tags
vocab= vocab_creator(tagged_titles_winner_df)

# Creating a dataframe from the dictionary
vocab_df = pd.DataFrame.from_dict(vocab,orient='index')

# Filling the nan values in the dataframe
vocab_df.fillna(value=0, inplace=True)

# Saving the top 5 most frequent NNP taggged words
size = 5
tag = 'NNP' 
top_proper_noun_nominee=vocab_df[tag].sort_values().tail(size)


# Plotting the top 5 most frequent NNP taggged words
title = 'Top {} Most Frequent Words for {} Tag'.format(size, tag)
top_proper_noun_nominee.plot(kind='barh', figsize=(12,6), title=title)
plt.show()


# --------------
""" After filling and submitting the feedback form, click the Submit button of the codeblock"""



