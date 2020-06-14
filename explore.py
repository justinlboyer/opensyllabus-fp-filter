#%%
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pd.set_option("max_colwidth", 999)

data_file = '/data/raw/matches.json'

root_pth = os.path.curdir
pth = root_pth + data_file

df = pd.read_json(pth, lines=True)

#%% Check counts of labels
data.groupby('label').count()

# %% See if there is any relationshop among labels and offsets
sns.violinplot(x='label', y='m1_start_offset'
, inner='quart'
, data=df)

# %%
sns.violinplot(x='label', y='m2_start_offset'
, inner='quart'
, data=df)

# %%
sns.violinplot(x='label', y='m1_end_offset'
, inner='quart'
, data=df)

# %%
sns.violinplot(x='label', y='m2_end_offset'
, inner='quart'
, data=df)


# Doesn't seem to be any relationship among offsets and labels, let's look at any linear relationships among the offsets

# %%
df_true = df.loc[df.label == True]

df_false = df.loc[df.label == False]

ax=sns.kdeplot(
    df_true['m1_start_offset']
    , df_true['m1_end_offset']
    , cmap="Reds"
)
ax=sns.kdeplot(
    df_false['m1_start_offset']
    , df_false['m1_end_offset']
    , cmap="Blues"
)

# %%
col1 = 'm1_start_offset'
col2 = 'm2_end_offset'
ax=sns.kdeplot(
    df_true[col1]
    , df_true[col2]
    , cmap="Reds"
)
ax=sns.kdeplot(
    df_false[col1]
    , df_false[col2]
    , cmap="Blues"
)

# %%
col1 = 'm1_end_offset'
col2 = 'm2_start_offset'
ax=sns.kdeplot(
    df_true[col1]
    , df_true[col2]
    , cmap="Reds"
)
ax=sns.kdeplot(
    df_false[col1]
    , df_false[col2]
    , cmap="Blues"
)

# %%
col1 = 'm1_end_offset'
col2 = 'm2_end_offset'
ax=sns.kdeplot(
    df_true[col1]
    , df_true[col2]
    , cmap="Reds"
)
ax=sns.kdeplot(
    df_false[col1]
    , df_false[col2]
    , cmap="Blues"
)

# No clear relationship among the offsets that we can use to our advantage
# %% Let's look at ngrams of the characters
n_gram = 3
df['author_grams'] = df['author'].apply(lambda x: [x[i:i+n_gram] for i in range(len(x)-n_gram+1)])

# %%

# %%
df['author_len'] = df.author.str.len()
sns.violinplot(x='label', y='author_len', data=df)

# %%

# %%
df['middle_len'] = df.middle.str.len()
sns.violinplot(x='label', y='middle_len', data=df)

print(df.loc[df['label']==True,'middle_len'].describe())

print(df.loc[df['label']==False,'middle_len'].describe())


# %%
df['title_len'] = df.title.str.len()
sns.violinplot(x='label', y='title_len', data=df)

print(df.loc[df['label']==True,'title_len'].describe())

print(df.loc[df['label']==False,'title_len'].describe())



# %%
def model(x, **kwargs):
    # think of a better way to set
    if 'middle_len_threshold' not in kwargs.keys():
        kwargs['middle_len_threshold'] = 13
    if 'title_len_threshold' not in kwargs.keys():
        kwargs['title_len_threshold'] = 0

    if x['middle_len'] >= kwargs['middle_len_threshold']:
        return False
    elif x['title_len'] <= kwargs['title_len_threshold']:
        return False
    else:
        return True

# %%
df['preds'] = df.apply(model, middle_len_threshold=0, axis=1)

# %%
print(classification_report(df['label'], df['preds']))

print(confusion_matrix(df['label'], df['preds']))

# %%
from sklearn.feature_extraction.text import CountVectorizer

train_df = df[df['split']=='train']
# %%

vec = CountVectorizer(analyzer='char_wb', ngram_range= (2,2))
counts = vec.fit_transform(train_df['author'])

# %%
vec.get_feature_names()

# %%
counts.toarray().shape

# %%
dev_df = df[df['split']=='dev']
test = vec.transform(dev_df['author'])


# %%
test.toarray().shape

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
tvec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,2))
tfidf = tvec.fit(train_df['author'])

# %%
cnt = tfidf.transform(train_df['author'])
cnt.toarray().shape

# %%
tfidf.get_feature_names()

# %%
import utils


# %%
vectorizer_paths = './models/'
col = 'middle'
vec = utils.load_vectorizers(os.path.join(vectorizer_paths, f'count_{col}.joblib'))
# %%
cols = vec.get_feature_names()
#%%
vecs = vec.transform(df[col])
#%%
vec_df = pd.DataFrame(vecs.todense(), columns=cols)
test = pd.concat([df, vec_df], axis =1)

# %%
df.count_middle_vec.toarray()

# %%
