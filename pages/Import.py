import streamlit as st
import pandas as pd
import numpy as np

"# :material/upload: Import data"

list_type=["Work shift","Pit","Topography","Block Model"]
data_type = st.radio("What data do you want to upload?", list_type)
