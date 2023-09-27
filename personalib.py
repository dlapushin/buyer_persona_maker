from io import StringIO
import streamlit as st
import pandas as pd
import os
import translate
from collections import Counter, OrderedDict
import pickle5 as pickle
import plotly.express as px
import plotly.graph_objects as go
import string
from stqdm import stqdm
stqdm.pandas()
import translate
import string
import re

import streamlit.components.v1 as components
from pivottablejs import pivot_ui


with open('title_classifier_serial', 'rb') as f:
    classifier = pickle.load(f)

with open('vectorizer_serial', 'rb') as f:
    vectorizer = pickle.load(f)

class Buyer_Persona:
    
    def __init__(self):
        self.df = None
        
    def csv_ingest(self, file_path, job_title_field):

        # Read the data from the given file_path into a pandas DataFrame
        # Assumes the file is either .csv or .xlsx
        if str(file_path.name).endswith(".csv"):
            print('Reading data file: ' + str(file_path.name))
            self.df = pd.read_csv(file_path, low_memory=False)
            self.df.drop(['Unnamed: 0'], axis=1, inplace=True)
        elif str(file_path.name).endswith(".xlsx"):
            self.df = pd.read_excel(file_path, low_memory=False)
            self.df.drop(['Unnamed: 0'], axis=1, inplace=True)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

        # Validate if the job_title_field exists in the DataFrame
        if job_title_field not in self.df.columns:
            raise ValueError(f"Job title field '{job_title_field}' not found in the DataFrame.")
    
    def salesforce_api_ingest(self, job_title_field):
        # use simplesalesforce library and send SOQL query to pull contacts, account, and opportunity data into a consolidated table
        
        pass

    def title_adjuster(self, job_title): ## removes extraneous punctuation

        title_adj = str.lower(str(job_title)).strip()
        title_adj = title_adj.replace('-',' ')
        title_adj = title_adj.translate(str.maketrans('', '', string.punctuation))
        title_adj = re.sub(r'\s+', ' ', title_adj)

        return title_adj
    
    def title_clean(self, job_title_field):
        if self.df is None:
            raise ValueError("DataFrame is empty. Please use data_ingest to load data first.")
        print('Cleaning data in field: ' + job_title_field)
        
        clean_title_field = str(job_title_field) + '_clean' ## creates new field based on the title field and adds "_clean"
        self.df[clean_title_field] = self.df[job_title_field].progress_apply(lambda x: self.title_adjuster(x))
        self.clean_title = clean_title_field
        
    def predict_category(self, token):
        token_features = vectorizer.transform([token])
        prediction = classifier.predict(token_features)

        return prediction[0]        
    
    ## Create a master dictionary of all the tokens found in titles
    ## Predict the POS (RES or FUN) for each token 
    ## Persist the resulting token:pos dictionary
    def master_token_to_odict(self):  
        self.series_title = self.df[self.clean_title].tolist()

        print('Flattening tokens into list format:')
        self.master_token = []
        for t in stqdm(self.series_title):
            token_split_list = t.split(' ')

            for t_elem in token_split_list:
                self.master_token.append(t_elem.strip())
                
        self.odict_sorted = OrderedDict(Counter(self.master_token).most_common())

        print('Creating dictionary of token to POS:')
        self.odict_pos = {}
        for key in stqdm(self.odict_sorted):
            self.odict_pos[key] = {'count': self.odict_sorted[key],
                            'pos': self.predict_category(key)
                           }

    def jt_tokenized_dict(job_title_text):

        df_job_title_breakdown = pd.DataFrame()
        dict_pos = {'RES': [],
                    'FUN': []
                   }

        for token in job_title_text.split(' '):
            predicted_pos = predict_category(token.lower().str.strip())

            if(predicted_pos == 'RES'):
                dict_pos['RES'].append(token)
            if(predicted_pos == 'FUN'):
                dict_pos['FUN'].append(token)

        dict_pos['RES'] = ' '.join(sorted(dict_pos['RES']))
        dict_pos['FUN'] = ' '.join(sorted(dict_pos['FUN']))
        
        return dict_pos


    def jt_tokenized_lists(self, input_title_clean):
        RES = []
        FUN = []

        for token in input_title_clean.split(' '):

            try:
                predicted_pos = self.odict_pos[token]['pos']
            except:
                predicted_pos = None

            if(predicted_pos == 'RES'):
                RES.append(token.strip())
            if(predicted_pos == 'FUN'):
                FUN.append(token.strip())
        
        return [RES, FUN]                

    def pos_mapper(self, **kwargs):
        
        # Apply the lambda function to the DataFrame
        print('Parsing field ' + self.clean_title + ' into RES and FUN components.  May take a few moments.')
        self.df = pd.concat([self.df, self.df.progress_apply(lambda x: self.jt_tokenized_lists(x[self.clean_title]), axis=1, result_type = 'expand')], axis=1)
        self.df.rename(columns={0:'RES', 1:'FUN'}, inplace=True)
        self.df['RES_flat'] = self.df['RES'].progress_apply(lambda x: ' '.join(x))
        self.df['RES_flat'] = self.df['RES_flat'].map(dict_res_synonyms).fillna(self.df['RES_flat'])
        
        self.df['FUN_flat'] = self.df['FUN'].progress_apply(lambda x: ' '.join(x))
        self.df['FUN_flat'] = self.df['FUN_flat'].map(dict_fun_synonyms).fillna(self.df['FUN_flat'])
        
        def won_list_flag(isclosed, iswon):
            if(isclosed == iswon == True):
                return 'Won'
            elif(isclosed == True and iswon == False):
                return 'Lost'
            else:
                return None

        if 'iswon' and 'isclosed' in kwargs:
            self.df['Won_Lost'] = self.df.apply(lambda x: won_list_flag(x[kwargs['isclosed']],x[kwargs['iswon']]), axis=1)

    ## CRITICAL FUNCTION: creates temporary dataframe (token_counts) which rank sorts either RES or FUN tokens.
    ## Returns a 2-column dataframe: POS and count
    def sorted_pos_token(self, pos):
        token_counts = pd.DataFrame(self.df.groupby(pos).size().sort_values(ascending=False))
        token_counts.reset_index(inplace=True)
        token_counts.columns = [pos, 'count']

        token_counts_filter = token_counts[token_counts[pos].isnull() == False]#token_counts.dropna() #[token_counts[pos].isna() == False]
        return token_counts_filter

    ## Calls the sorted_pos_token function and transforms the sorted RES_flat and FUN_flat tokens into lists.
    ## Makes POS token lookups more efficient
    def pos_sorter(self):
        print('Sorting RES and FUN values')
        self.df_res = self.sorted_pos_token('RES_flat')[['RES_flat','count']]
        self.list_res = list(self.df_res['RES_flat'])
        self.list_res.remove('')

        self.df_fun = self.sorted_pos_token('FUN_flat')[['FUN_flat','count']]                           
        self.list_fun = list(self.df_fun['FUN_flat'])
        self.list_fun.remove('')

    ## Futher reduces the POS list lookup set to only the number of combinations given by "n" (e.g 20x20)        
    def top_pos_sorter(self, n):
        print('Finding top RES and FUN values')
 
        self.top_res = self.list_res[0:n]
        self.top_fun = self.list_fun[0:n]
        
    # You would call top_pos_match separately with closed-won then closed-lost parameter defs
    # This will return the dataframe df_top_pos_matches
    # Thinking this should be returned explicitly and not as a self attribute.
    def top_pos_match(self, closed_status, won_status):

        filter_res = (self.df['RES_flat'].isin(self.top_res))
        filter_fun = (self.df['FUN_flat'].isin(self.top_fun))
        filter_closed_status = (self.df['IsClosed'] == closed_status)
        filter_won_status = (self.df['IsWon'] == won_status)
        
        sorter_res_Index = dict(zip(self.top_res, range(len(self.top_res))))
        sorter_fun_Index = dict(zip(self.top_fun, range(len(self.top_fun))))
        df_top_pos_matches = self.df[filter_res & filter_fun & filter_closed_status & filter_won_status]  
        df_top_pos_matches['RES_sortrank'] = df_top_pos_matches['RES_flat'].map(sorter_res_Index)
        df_top_pos_matches['FUN_sortrank'] = df_top_pos_matches['FUN_flat'].map(sorter_fun_Index)
        df_top_pos_matches.sort_values(by=['RES_sortrank'], ascending=True, inplace=True)
        df_top_pos_matches.reset_index(inplace=True, drop=True)
        
        df_top_pos_pivot = df_top_pos_matches.groupby(['RES_flat','FUN_flat']).count().reset_index()[['RES_flat','FUN_flat','Title']]
        df_top_pos_pivot_smry = df_top_pos_pivot.pivot(index='RES_flat', columns='FUN_flat')['Title'].fillna(0)
        
        return df_top_pos_pivot_smry   
        

dict_res_synonyms = {
'senior architect':'architect',
'principal architect':'architect',
'adviser':'advisor/consultant',
'advisor':'advisor/consultant',
'advisors':'advisor/consultant',
'advisory':'advisor/consultant',
'counsel':'advisor/consultant',
'counselor':'advisor/consultant',
'expert':'advisor/consultant',
'lawyer':'advisor/consultant',
'attorney':'advisor/consultant',
'senior consultant':'advisor/consultant',
'consultant':'advisor/consultant',
'senior specialist':'advisor/consultant',
'specialist':'advisor/consultant',
'sr analyst':'analyst',
'senior analyst':'analyst',
'assistant':'assistant',
'assitant':'assistant',
'asst':'assistant',
'assoc':'associate',
'associate':'associate',
'associates':'associate',
'contractor':'contracting',
'contracts':'contracting',
'direct':'director',
'associate director':'director',
'director':'director',
'director management':'director',
'director general':'director',
'directors':'director',
'developer':'engineer',
'lead developer':'engineer',
'devops engineer':'engineer',
'engineer':'engineer',
'lead engineer':'engineer',
'principal engineer':'engineer',
'sr engineer':'engineer',
'senior developer':'engineer',
'sr developer':'engineer',    
'senior devops engineer':'engineer',
'programmer':'engineer',
'technologist':'engineer',
'ceo':'executive',
'cfo':'executive',
'chair':'executive',
'chairman':'executive',
'chief':'executive',
'chief officer':'executive',
'cio':'executive',
'ciso':'executive',
'cmo':'executive',
'cofounder':'executive',
'commander':'executive',
'commissioner':'executive',
'coo':'executive',
'cso':'executive',
'cto':'executive',
'evp':'executive',
'exec':'executive',
'founder':'executive',
'founding':'executive',
'head':'executive',
'head management':'executive',
'owner':'executive',
'partner':'executive',
'president':'executive',
'senior executive':'executive',    
'administrator':'manager',
'lead':'manager',
'leader':'manager',
'managed':'manager',
'management':'manager',
'manager management':'manager',
'manager':'manager',
'managing':'manager',
'managment':'manager',
'manger':'manager',
'mgmt':'manager',
'mgr':'manager',
'principal':'manager',
'senior engineer':'engineer',
'sr developer':'engineer',
'senior director':'director',
'sr director':'director',
'sr manager':'manager',
'senior manager':'manager',
'superintendent':'manager',
'supervisor':'manager',
'avp':'vice-president',
'svp':'vice-president',
'vice':'vice-president',
'vp':'vice-president',
'vice president':'vice-president',
'sr':'senior',
'regional vice president':'vice-president'
}

dict_fun_synonyms = {
'it':'information technology',
'it operations':'information technology',
'information':'information technology',
'technology':'information technology',
'service':'services',
'system':'systems',
'solution':'solutions',
'qa':'quality assurance',
'application':'software',    
'application development':'software',     
'business systems':'business',    
'application':'application development',    
'quality assurance':'quality assurance',
'software engineering':'software',    
'salesforce':'crm'
}