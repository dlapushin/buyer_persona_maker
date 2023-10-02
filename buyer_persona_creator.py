import streamlit as st
from st_aggrid import AgGrid
import personalib as perlib
from stqdm import stqdm

import pickle5 as pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

from pivottablejs import pivot_ui
import pandas as pd
from datetime import datetime   
from io import StringIO
from urllib.error import URLError
import datetime

with open('title_classifier_serial', 'rb') as f:
    classifier = pickle.load(f)

with open('vectorizer_serial', 'rb') as f:
    vectorizer = pickle.load(f)

df_res_synonyms = pd.DataFrame.from_dict(perlib.dict_res_synonyms, orient='index', columns=['res']).sort_values(by='res')
df_res_synonyms.reset_index(inplace=True)
df_res_synonyms.columns = ['Token', 'Mapped to RES']

df_fun_synonyms = pd.DataFrame.from_dict(perlib.dict_fun_synonyms, orient='index', columns=['fun']).sort_values(by='fun')
df_fun_synonyms.reset_index(inplace=True)
df_fun_synonyms.columns = ['Token', 'Mapped to FUN']

st.set_page_config(layout="wide")

def intro(approach):

    st.header('Buyer Persona Creation Tools for :blue[Sales and Marketing]')
    st.divider()
    st.sidebar.success("Select approach above.")

    st.markdown(
        """
        Helps streamline the creation of Sales and Marketing
        Buyer Persona in a fully data-driven way.

        **👈 Select an approach from the dropdown on the left** to begin.
    """
    )

def contacts_import(approach):


    res_fun_topN = None

    uploaded_file = st.sidebar.file_uploader("Choose a file", 
                                        type=['csv','xlsx']
                                        )     


    def timestamp_str_to_datetime(ts_string):

        ts_date = datetime.datetime.strptime(ts_string[0:10], "%Y-%m-%d")
        return ts_date

    with st.container():

        global buyer_persona
        import datetime

        #@st.cache
        def file_import(file, title_field):
            #st.session_state.clicked = False
            buyer_persona = perlib.Buyer_Persona()
            buyer_persona.csv_ingest(file, title_field)
            return buyer_persona

        if(approach == 'Simple approach'):
            st.header('Buyer Persona Creation Tools for :blue[Sales and Marketing]')
            st.divider()
            st.subheader("Simple Approach")
            st.write(
                """
                Upload a file with **Contact ID, Contact Job Title, Contact CreatedDate** at a minimum.
                **No personal information (PII) is required**.
                """
            )

            if(uploaded_file is not None):
                                
                if "uploaded" not in st.session_state:
                    st.session_state.uploaded = 'simple'

                buyer_persona = file_import(uploaded_file, 'Title')

                col1_fldid, col2_fldid = st.columns(2)

                with col1_fldid:

                    option_contact_createdate = st.selectbox(
                        'Field for Contact CreatedDate',
                        buyer_persona.df.columns)

                    option_contact_title = st.selectbox(
                        'Field for Contact Job Title',
                        buyer_persona.df.columns)

                with col2_fldid:                

                    try:
                        date_range_filter = st.slider('Select date range', 
                                         value = (timestamp_str_to_datetime(min(buyer_persona.df[option_contact_createdate].dropna())),
                                            timestamp_str_to_datetime(max(buyer_persona.df[option_contact_createdate].dropna()))),
                                         key='date_slide'
                                         ) 
                    except:
                        st.stop()

                    ## NEED TO MEMOIZE the original DF and make filtered copies
                    buyer_persona.df = buyer_persona.df

                field_selections = [option_contact_title,
                                option_contact_createdate]

                if date_range_filter:
                    st.session_state.date_range = date_range_filter

                if 'clicked' not in st.session_state:
                    st.session_state.clicked = False

                def click_button():
                    st.session_state.clicked = True

                st.button("Process data", 
                            type="primary", 
                            on_click=click_button)

                                
        if(approach == 'Advanced approach'):
            
            st.header('A Buyer Persona Toolkit for :blue[Sales and Marketing]')
            st.subheader("Advanced Approach")

            st.write(
                """
                Upload a file with **Contact ID, Contact Job Title, Contact CreatedDate** 
                """
                )
            st.write(
                """
                PLUS: **Opportunity details (e.g. Opportunity Amount, CreatedDate, CloseDate, IsClosed, IsWon, 
                and GTM segment details**
                **No personal information (PII) is required**.
                """
            )

            if(uploaded_file is not None):
                file_import(uploaded_file, 'Title')

            col1_fldid, col2_fldid, col3_fldid = st.columns(3)

            with col1_fldid:

                option_contact_title = st.selectbox(
                    'Field for Contact Job Title',
                    buyer_persona.df.columns)

                option_contact_createdate = st.selectbox(
                    'Field for Contact CreatedDate',
                    buyer_persona.df.columns)

                option_oppty_createddate = st.selectbox(
                    'Field for Opportunity CreatedDate',
                    buyer_persona.df.columns)

            with col2_fldid:

                option_oppty_amount = st.selectbox(
                    'Field for Opportunity Amount',
                    buyer_persona.df.columns)

                option_oppty_closedate = st.selectbox(
                    'Field for Opportunity CloseDate',
                    buyer_persona.df.columns)

                option_oppty_isclosed = st.selectbox(
                    'Field for Opportunity IsClosed',
                    buyer_persona.df.columns)

                option_oppty_iswon = st.selectbox(
                    'Field for Opportunity IsWon',
                    buyer_persona.df.columns)

            with col3_fldid:

                option_oppty_gtmseg = st.selectbox(
                    'Field for Opportunity GTM Segment',
                    buyer_persona.df.columns)

            field_selections = [option_contact_title,
                                option_contact_createdate,
                                option_oppty_amount,
                                option_oppty_createddate,
                                option_oppty_closedate,
                                option_oppty_isclosed,
                                option_oppty_iswon,
                                option_oppty_gtmseg]

        def get_contacts_data(*args):
            
            res_fun_topN = st.sidebar.slider('Select value for top N', 
                                     min_value=2, 
                                     max_value=20,
                                     value = 10, 
                                     key='slide'
                                     ) 

            if res_fun_topN:
                st.session_state.res_fun_topN = int(res_fun_topN)

            st.write('File loaded, data being processed.')
           
            ### CODE FOR PROCESSING FILE DATA
            buyer_persona.title_clean(option_contact_title)
            buyer_persona.master_token_to_odict()  ## makes dictionary for all unique tokens found in the titles

            if approach == 'Simple approach':

                buyer_persona.pos_mapper()
                buyer_persona.df[option_contact_createdate] = pd.to_datetime(buyer_persona.df[option_contact_createdate], infer_datetime_format=True) 

            if approach == 'Advanced approach':

                buyer_persona.pos_mapper(isclosed = option_oppty_isclosed,
                                         iswon = option_oppty_iswon)

            buyer_persona.pos_sorter()

            ### DISPLAY TOP TOKENS FOR RES AND FUN

            st.write('### Top {0} RES and FUN tokens: '.format(int(res_fun_topN)))

            ## CODE TO DISPLAY TOP RES AND FUN ARRAYS IN 2 COLUMNS
            col1, col2 = st.columns(2)

            with col1:
                top_res = buyer_persona.df_res.query('RES_flat != \'\'')
                top_res.reset_index(inplace=True, drop=True)
                st.table(top_res.iloc[0:res_fun_topN]) 
                top_res_tokens = list(top_res['RES_flat'][0:res_fun_topN])

            with col2:
                top_fun = buyer_persona.df_fun.query('FUN_flat != \'\'')
                top_fun.reset_index(inplace=True, drop=True)
                st.table(top_fun.iloc[0:res_fun_topN]) 
                top_fun_tokens = list(top_fun['FUN_flat'][0:res_fun_topN])

            st.session_state['buyer_persona'] = buyer_persona

            st.divider()
            st.subheader('Persona Analysis')
            st.write('**Most prevalent personas in upper left of Heatmap below**')
            st.write('Tokens under **Mappings** tab have been mapped and consolidated for semantic consistency')

            pivot_tab, mapping_tab = st.tabs(["📈 Heatmap", "🗃 Mappings"])

            with pivot_tab:

                res_fun_choice, coverage_metric = st.columns(2)

                with res_fun_choice:
                    match_choice = st.radio("Choose RES-FUN matching approach using above table",
                        options=['Top RES x Top FUN', 
                                 'Top RES x All FUN', 
                                 'All RES x Top FUN'],
                        )


                ## DATE FILTER DEFINITIONS
                min_dt = pd.to_datetime(date_range_filter[0], utc=True)
                max_dt = pd.to_datetime(date_range_filter[1], utc=True)
                filter_create_date = (buyer_persona.df[option_contact_createdate].between(min_dt,max_dt))

                ## FILTER DEFINITIONS

                filter_res = (buyer_persona.df['RES_flat'].isin(top_res_tokens))
                filter_fun = (buyer_persona.df['FUN_flat'].isin(top_fun_tokens))
                                
                if(match_choice == 'Top RES x Top FUN'):

                    pivot_df = buyer_persona.df[filter_res & filter_fun & filter_create_date]
                    pivot_df_agg = pivot_df.groupby(['RES_flat','FUN_flat']).count()
                    pivot_df_agg.reset_index(inplace=True)

                elif(match_choice == 'Top RES x All FUN'):

                    pivot_df = buyer_persona.df[filter_res & filter_create_date]
                    pivot_df_agg = pivot_df.groupby(['RES_flat','FUN_flat']).count().sort_values(by=option_contact_createdate, ascending=False)
                    pivot_df_agg.reset_index(inplace=True)

                elif(match_choice == 'All RES x Top FUN'):

                    pivot_df = buyer_persona.df[filter_fun & filter_create_date]
                    pivot_df_agg = pivot_df.groupby(['RES_flat','FUN_flat']).count().sort_values(by=option_contact_createdate, ascending=False)
                    pivot_df_agg.reset_index(inplace=True)

                if approach == 'Simple approach':
                    field_filter = ['RES_flat','FUN_flat']
                elif approach == 'Advanced approach':
                    field_filter = ['RES_flat','FUN_flat','Won_Lost']


                t = pivot_ui(pivot_df[field_filter], 
                            rows=['RES_flat'], 
                            cols=['FUN_flat'],
                            rendererName = 'Heatmap',
                            rowOrder = 'value_z_to_a', 
                            colOrder = 'value_z_to_a',
                            )

                
                total_contact_count = buyer_persona.df.shape[0]
                total_persona_match = pivot_df.shape[0]

                with coverage_metric:
                    st.metric(label="Coverage rate (% of contacts mapped)", value=str(round(100*total_persona_match/total_contact_count))+'%')

                with open(t.src) as t:
                    
                    try:
                        components.html(t.read(), width=1200, height=400, scrolling=True)
                        st.subheader('Underlying Records')
                        AgGrid(pivot_df)
                    except:
                        st.write('Dataset aggregation too granular. Try reducing value for N')
            
            with mapping_tab:

                col_res_syn, col_fun_syn = st.columns(2)

                with col_res_syn:
                    st.dataframe(df_res_synonyms)
                with col_fun_syn:
                    st.dataframe(df_fun_synonyms)


        if uploaded_file:
            if(approach == 'Simple approach' and st.session_state.clicked == True):
                get_contacts_data()
            elif approach == 'Advanced approach':
                get_contacts_data(option_oppty_isclosed,option_oppty_iswon)        
        else:
            pass

page_names_to_funcs = {
    "—": intro,
    "Simple approach": contacts_import,
    #"Advanced approach": contacts_import
}

app_name = st.sidebar.selectbox("Choose a function", page_names_to_funcs.keys())
page_names_to_funcs[app_name](app_name)                                    

