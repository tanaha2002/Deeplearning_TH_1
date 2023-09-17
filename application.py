import streamlit as st
import pandas as pd
import numpy as np
import pickle

#set application always wide mode
st.set_page_config(layout="wide")

st.title('Classification Model')
#list model load
models = ['xg_boost.pkl']


col1, col2, col3 = st.columns([0.5,1.65,2.25])
def header():
    with col2:
        st.markdown('###### 1. Select Input Type')
        global Input_type 
        Input_type= st.selectbox('Input Type?', ('User Input', 'Loading File'))
    with col3:
        global col3_1, col3_2 
        col3_1,col3_2 = st.columns(2)
        with col3_1:
            st.markdown('###### 2. Select Model')
            global Model_type 
            Model_type = st.selectbox(
                'Model Select?',
                (models))
        if Input_type == 'User Input':
            with col3_2:
                st.markdown('###### 3. Predict')   
        
        




def user_input():
    global col1_1, col1_2
    col1_1, col1_2 = st.columns(2)
    
    with col1_1:
        feature_1 = st.slider('attribute 1', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        feature_2 = st.slider('attribute 2', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        feature_3 = st.slider('attribute 3', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        feature_4 = st.slider('attribute 4', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        feature_5 = st.slider('attribute 5', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        feature_6 = st.slider('attribute 6', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        feature_7 = st.slider('attribute 7', min_value=0.0, max_value=20.0, value=5.0, step=0.1)

    return pd.DataFrame([[feature_1, feature_2, feature_3, feature_4,feature_5,feature_6,feature_7]], columns=['attribute 1', 'attribute 2', 'attribute 3', 'attribute 4', 'attribute 5', 'attribute 6','attribute 7'])



    





def loading_file():
    global upload_file
    with col2:
        upload_file = st.file_uploader('Upload File', type=['csv'])
        if upload_file is not None:
            df = pd.read_csv(upload_file,header=None, names=['attribute 1', 'attribute 2', 'attribute 3', 'attribute 4', 'attribute 5', 'attribute 6','attribute 7'])
            return df
        else:
            return None
    







    

    

def predict(model,new_data):
        #remove feature not use
        new_data = new_data.drop(['attribute 6'], axis=1)
        predict_labels = model.predict(new_data)
        st.write('Predicted Labels: {}'.format(predict_labels+1))



def predict_file(model,data):
        #remove feature not use
        data = data.drop(['attribute 6'], axis=1)
        predict_labels = model.predict(data)

        show_data_each_row(data,model,data,predict_labels)



def load_model():
    with open(Model_type, 'rb') as file:
        #load xg_boost.pkl
        model = pickle.load(file)

    return model





        







def show_data_each_row(df,model,new_data,predict_labels):
    

    new_data['Predicted Labels'] = predict_labels+1
    for i in range(len(new_data)):
        #margin left css
        with col2:
            st.write(new_data.iloc[i:i+1])
            st.write('#')
            st.write('#')





        
def week1():
    header()
    if Input_type == 'User Input':
        col1, col2, col3 = st.columns([0.5,1.25,2.25])
        with col2:
            new_data = user_input()

        with col3_2:
            model = load_model()
            st.write('#')
            predict(model,new_data)
    elif Input_type == 'Loading File':
        col1, col2 = st.columns([1.25,0.25])
        with col2:
            data = loading_file()
        if data is not None:
            model = load_model()
            predict_file(model,data)
            



        
        





#main
week1()








    