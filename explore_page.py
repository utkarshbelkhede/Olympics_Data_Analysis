import streamlit as st
import pandas as pd
import seaborn as sns


def explore():
    st.title("Explore")

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write("""
    ### Loading Olympics Dataset into DataFrame
    """)

    events = pd.read_csv("./Datasets/athlete_events.csv")
    st.dataframe(events.head())

    rows = events.shape[0]
    cols = events.shape[1]

    st.write("The above Olympics Dataset has", rows ," and", cols, "columns.")

    st.write("""
    ### Loading NOC Regions Dataset into DataFrame
    """)

    regions = pd.read_csv("./Datasets/noc_regions.csv")
    st.dataframe(regions.head())

    rows = regions.shape[0]
    cols = regions.shape[1]

    st.write("The above Olympics Dataset has", rows, " and", cols, "columns.")

    st.write("""
                ### Select Season
            """)

    options = (
        'Both',
        'Summer',
        'Winter'
    )

    season = st.radio('', options)

    if season == "Both":
        df = events
    else:
        df = events[events['Season'] == season]

    null_features = [[features, df[features].isnull().sum()] for features in df.columns if
                     df[features].isnull().sum() > 0]

    null_features = pd.DataFrame(null_features, columns=['Feature', 'No. of Null Values'])

    st.write(
        """
    ### Null Values in Dataset
    """
    )
    st.dataframe(null_features)

    st.write("""
            ### Heatmap of Missing Values
            """)

    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    fig = sns.set(rc={'figure.figsize': (14, 10)})
    st.pyplot(fig)