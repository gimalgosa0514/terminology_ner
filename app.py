# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:21:43 2024

@author: gimal
"""

import streamlit as st

from inference import inference_fn


result = ""
with open("result.txt","r") as f:
    for line in f:
        result += line
        
#제목
st.title("🤗BERT-NER")

#탭 생성
tab1, tab2 = st.tabs(["inference", "evaluation"])

#inference 탭
with tab1:
    #컨테이너를 만들어서 공간을 분리
    with st.container(border=True):
        st.text_input("예측할 문장을 입력 후 Enter 키를 누르세요", key="sentence")
    
    #문자열임 이걸로 inference 하면 됨.
    sentence = st.session_state.sentence
    st.table(inference_fn(sentence))

#evaluation 결과 탭 
with tab2:
    st.write("채점 결과")
    with st.container(border=True):
        st.text(result)