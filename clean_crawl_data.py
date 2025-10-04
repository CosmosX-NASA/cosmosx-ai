import os
import json
import pandas as pd
from typing import List, Dict


def format_caption(row) -> List[Dict[str, str]]:
    try:
        if not row['figures_json']:
            return row
        data_list = json.loads(row['figures_json'])
        results = [{
            'url': data['images'][0] if len(data['images']) > 0 else '',
            'caption': data['label'] if len(data['label']) != 0 else data['caption']
        } for data in data_list]
        row['figures_json'] = json.dumps(results)
        return row
    except:
        print(row['PMCID'])
        row['figures_json'] = None
        return row


df = pd.read_csv('info.csv')
df = df.dropna().drop_duplicates()
# df = df.drop(columns=['figure_image_urls', 'figure_captions'])

df = df.apply(lambda row: format_caption(row), axis=1)
df = df.dropna()

df = pd.read_csv('info.csv')
col_list = list(set(df.loc[:, 'PMCID'].to_list()))
# paper_list = [paper.split('.')[0] for paper in [
#     file for file in os.listdir('papers') if file.endswith('.md')]]

# for paper in paper_list:
#     if paper not in col_list:
#         os.remove(os.path.join('papers', f"{paper}.md"))

paper_list = [paper.split('.')[0] for paper in [
    file for file in os.listdir('papers') if file.endswith('.md')]]
print(len(paper_list))
delete_list = []
for col in col_list:
    if col not in paper_list:
        delete_list.append(col)

df = df[df['PMCID'].apply(
    lambda pmcid: pmcid not in delete_list)]
len(df)
# df.to_csv(os.path.join(main_path, 'info.csv'), index=None)

df.drop(columns='figures_json').reset_index(
    drop=True).to_csv('info.csv', index=None)

clean_keyword = {}

for file in [file for file in os.listdir('a') if file.endswith('.csv')]:
    keyword_df = pd.read_csv(f'a/{file}')
    pmcid_list = keyword_df.loc[:, 'PMCID'].to_list()
    keywords_list = keyword_df.loc[:, 'Clean_Keywords'].to_list()
    for i in range(len(pmcid_list)):
        clean_keyword[pmcid_list[i]] = keywords_list[i]

df['keywords'] = ''
mapping = {str(k): v for k, v in clean_keyword.items()}
df['keywords'] = df['PMCID'].astype(str).map(mapping).fillna(df['keywords'])
df['keywords'] = df.loc[:, 'keywords'].str.replace(' / ', ',')
# df.to_parquet('info.parquet')

for index, row in df.iterrows():
    figures_json = json.loads(row['figures_json'])
    save = []
    for fig in figures_json:
        if fig['url'] == '' or fig['caption'] == '':
            continue
        save.append(fig)
    row['figures_json'] = json.dumps(save)

df.reset_index(drop=True).to_parquet('info.parquet')

pd.read_parquet('info.parquet')
