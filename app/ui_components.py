import streamlit as st
import plotly.express as px
import pandas as pd

def display_domain_treemap(domain_contributions: dict):
    """
    Displays an interactive treemap of domain contributions using Plotly.

    Args:
        domain_contributions (dict): A dictionary where keys are domain names
                                     and values are contribution percentages.
    """
    if not domain_contributions or sum(domain_contributions.values()) == 0:
        st.info("No specific domain contributions were identified for this answer.")
        return

    df = pd.DataFrame(list(domain_contributions.items()), columns=['Domain', 'Contribution'])
    df = df[df['Contribution'] > 0]

    fig = px.treemap(
        df,
        path=[px.Constant("Domain Contributions"), 'Domain'],
        values='Contribution',
        color='Contribution',
        color_continuous_scale='YlGnBu',
        hover_data={'Contribution': ':.2f%'}
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), font=dict(size=16))
    st.plotly_chart(fig, use_container_width=True)