import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
pages_dict = {
  "HOME"   : [st.Page("pages/Home.py", title="Home", icon=":material/elderly_woman:")],
  "IMPORT" : [st.Page("pages/Import.py", title="Import files", icon=":material/upload:")],
}

pg = st.navigation( pages_dict, position="top" )

pg.run()
