import streamlit as st
import numpy as np
import plotly.express as px
import engine
import plotly.graph_objects as go
import altair as alt
import requests
import tempfile
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration, TapasTokenizer, TapasForQuestionAnswering, AutoTokenizer, AutoModelForSeq2SeqLM
import torch



from pyspark.sql import SparkSession

### ------------------ CACHED SETUP ------------------

@st.cache_resource
def get_spark_session():
    return SparkSession.builder.appName("Museh PySpark Learning").getOrCreate()

@st.cache_resource
def load_data():
    return spark.read.json('/Users/angel/Downloads/spring25data/app/listen_events')

@st.cache_resource
def get_clean_data():
    return engine.clean(df=load_data())

@st.cache_data
def get_artist_list(df, threshold=1000):
    return engine.get_artist_over_1000(df=df, number_of_lis=threshold)

@st.cache_data
def get_top_artists_by_state(_df, state):
    return engine.get_top_10_artists(df=_df, state=state)

@st.cache_resource
def get_map_data(_df, artist):
    #artist_df = e.get_artist_state_listen(df=_df, artist=artist)
    return engine.get_artist_state(df=_df,artist=artist)

@st.cache_data
def kpis(_df):
    return engine.calculate_kpis(df=_df)

@st.cache_data
def user_list(_df, state):
    return engine.get_user_list(df = _df, state=state)

@st.cache_data
def top_paid(_df, state):
    return engine.top_paid_songs(df=_df, state=state)

@st.cache_data
def top_free(_df, state):
    return engine.top_free_songs(df=_df, state=state)

@st.cache_data
def create_pie(_df, state):
    return engine.create_subscription_pie_chart(df=_df, state=state)

@st.cache_data
def build_prommpt_from_dataframe(_df):
    return engine.build_prompt_from_dataframe(df=_df)

@st.cache_data
def get_bottom_3_artists(_df):
    return engine.get_bottom_3_artists(df=_df)

@st.cache_data
def build_prompt_from_bottom_artists(_df):
    return engine.build_prompt_from_bottom_artists(df=_df)

#load t5-small model
@st.cache_resource
def load_flan_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

#load flan-t5-small model
@st.cache_resource
def load_t5_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cpu()
    return tokenizer, model

### ------------------ INITIAL STATE ------------------

if "option" not in st.session_state:
    st.session_state.option = "Kings Of Leon"

if "location" not in st.session_state:
    st.session_state.location = "Nationwide"

### ------------------ PAGE CONFIG ------------------
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'><span style='color: white'>Muse</span><span style='color: #87CEEB;'>Dash</span></h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Pipeline", "Dashboard", "Repo"])

spark = get_spark_session()
clean_listen = get_clean_data()

### ------------------ MAP RENDER FUNCTION ------------------

def render_map(artist):
    c = get_map_data(clean_listen, artist)
    

    fig = go.Figure(data=go.Choropleth(
        locations=c.state,
        z=c.listens,
        locationmode='USA-states',
        colorscale='Blues',
        colorbar_title="Number of\n Listens"
    ))

    fig.update_layout(
        
        geo_scope='usa',
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    event = st.plotly_chart(fig, on_select="rerun", selection_mode=["points", "box", "lasso"])
    points = event["selection"].get("points", [])

    if points:
        selected_state = points[0]["location"]
        if selected_state != st.session_state.location:
            st.session_state.location = selected_state
            st.rerun()
    else:
        # If background is clicked (no state), reset to Nationwide
        if st.session_state.location != "Nationwide":
            st.session_state.location = "Nationwide"
            st.rerun()

        try:
            tokenizer, model = load_t5_model()

            def table_to_text(c, question):
                """
                Dynamically generate context based on the question.
                """
                if "most listens" in question.lower():
                    # Focus on the state with the most listens
                    top_state = c.loc[c["listens"].idxmax()]
                    return f"In {top_state['state']}, {top_state['artist']} had {top_state['listens']} listens."
                elif "least listens" in question.lower():
                    # Focus on the state with the least listens
                    least_state = c.loc[c["listens"].idxmin()]
                    return f"In {least_state['state']}, {least_state['artist']} had {least_state['listens']} listens."
                elif "top artist" in question.lower():
                    # Focus on the top artist in the selected state
                    top_artist = c.loc[c["listens"].idxmax()]
                    return f"The top artist in {top_artist['state']} is {top_artist['artist']} with {top_artist['listens']} listens."
                elif "bottom artist" in question.lower():
                    # Focus on the bottom artist in the selected state
                    bottom_artist = c.loc[c["listens"].idxmin()]
                    return f"The bottom artist in {bottom_artist['state']} is {bottom_artist['artist']} with {bottom_artist['listens']} listens."
                elif "total listens" in question.lower():
                    # Focus on the total listens across all states
                    total_listens = c["listens"].sum()
                    return f"The total listens across all states is {total_listens}."
                else:
                    # Default: summarize the entire table
                    rows = [f"In {row['state']}, {row['artist']} had {row['listens']} listens." for _, row in c.iterrows()]
                    return " ".join(rows)
            question = st.text_input("Enter your question:", "Which state had the most listens?")
            if question:
                try:
                    with st.spinner("Generating answer..."):
                        context = table_to_text(c, question)
                        prompt = f"""
                                    Table: {context}
                                    Question: {question}
                                    Answer:
                                    """
                        inputs = tokenizer(prompt.strip(), return_tensors="pt", truncation=True, max_length=600)
                        outputs = model.generate(**inputs, max_new_tokens=50)
                        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        st.markdown(f"**Answer:** {answer}")
                        st.markdown("<p style='font-size: 0.85em; color: gray;'>AI-generated answer</p>", unsafe_allow_html=True)         

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
        except Exception as e:
            st.error(f"Error loading T5 model: {e}")
    

### ------------------ MAIN UI: TAB 1 ------------------


### ------------------ MAIN UI: TAB 2 ------------------
with tab2:
    # setting dashboard column layout
    col_table = st.columns((5, 10), gap='medium')    

    with st.container(border=True):

        with col_table[0]:
            with st.container(border=True):
                # creating top 10 chart dataframe
                top_10 = get_top_artists_by_state(clean_listen, st.session_state.location)
                state_text = st.session_state.location if st.session_state.location != "Nationwide" else "the Nation"
                st.header(f"Top 10 Artists in {state_text}")
                
                # creates the box to pic artist
                selected_row = st.dataframe(
                    top_10,
                    use_container_width=True,
                    selection_mode="single-row",
                    on_select="rerun",
                    hide_index=True
                )

                rows = selected_row['selection'].get("rows", [])
                if rows:
                    selected_artist = top_10.Artist[rows[0]]
                    if selected_artist != st.session_state.option:
                        st.session_state.option = selected_artist
                        st.rerun()
            
                try:
                    # Load model after Spark and data prep
                        tokenizer, model = load_flan_model()

                        bottom_3 = get_bottom_3_artists(clean_listen)

                        prompt_text = engine.build_prompt_from_bottom_3(df=bottom_3)
                        
                        # prompt_text = engine.build_prompt_from_top10(df=bottom_artists_df)
                        # Generate summary prompt
                        if not bottom_3.empty:
                            prompt_text = engine.build_prompt_from_bottom_3(bottom_3)
                            with st.spinner("Generating summary..."):   
                                # Tokenize and generate summary
                                inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
                                outputs = model.generate(**inputs, max_length=100, min_length=30, do_sample=False, top_p=0.95, top_k=50, num_beams=3)
                                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

                                def capitalize_sentences(text):
                                    sentences = text.split('. ')
                                    capitalized_sentences = []
                                    for sentence in sentences:
                                        words = sentence.split()
                                        capitalized_words = [word.capitalize() if word.islower() else word for word in words]
                                        capitalized_sentence = ' '.join(capitalized_words)
                                        capitalized_sentences.append(capitalized_sentence)
                                    return '. '.join(capitalized_sentences)
                                
                                summary = capitalize_sentences(summary)
                                st.write(summary)
                                st.markdown("<p style='font-size: 0.85em; color: gray;'>AI-generated summary</p>", unsafe_allow_html=True)
                        else:
                            st.info("No data available to summarize for selected filters.")
                except Exception as e:
                    st.error(f"Error loading model or generating summary: {e}")
                    st.write("Please check your model and data.")
        
            with st.container():
                # KPI metrics
                kpi_data = kpis(_df=clean_listen)
                col1, col2, col3 = st.columns([1.5, 2, 2.2])
                with col1:
                    with st.container(border=True):
                        st.metric("Total Users", f'{round(kpi_data[0]/1000)}k+')
                with col2:
                    with st.container(border=True):
                        st.metric("Average Total Listening", f"{round(kpi_data[1]/60)} MIN")
                with col3:
                    with st.container(border=True):
                        st.metric("Total Paid Listening", f"{round(kpi_data[2]/3600000)}k+ H")
            
            with st.container(border=True):
               
                # listen graph creation
                listen_duration = user_list(_df=clean_listen, state=st.session_state.location)
                chart_state = st.session_state.location if st.session_state.location != "Nationwide" else "the Nation"
                
                #create the line graph
                line_fig = px.line(
                    listen_duration,
                    x="month_name",
                    y="total_duration",
                    color="subscription",
                    title= f'How long are users listening in {chart_state}',
                    labels={"month_name": "Month", "total_duration": "Total Duration (seconds)"}
                        )
                
                
                line_fig.update_layout(
                    hovermode="x unified",

                    #style the hover line color
                    xaxis=dict(
                        showspikes=True,
                        spikemode='across',
                        spikesnap='cursor',
                        spikethickness=1,
                        spikecolor="lightgray"
                    ),
                    yaxis=dict(
                        showspikes=False, #turns off horizontal line

                    )
                                        )

                # Update hovertemplate for the 'Paid' trace
                line_fig.update_traces(
                    selector={'name': 'paid'},
                    hovertemplate='<span style="font-size: 18px;">' +
                                'Paid: %{y:.2f}' +
                                '<extra></extra>'
                    )

                # Update hovertemplate for the 'Free' trace
                line_fig.update_traces(
                    selector={'name': 'free'},
                    hovertemplate='<span style="font-size: 18px;">' +
                                'Free: %{y:.2f}' +
                                '<extra></extra>',
        
                )


                 #change color of the lines
                line_fig.update_traces(
                selector={'name': 'paid'},
                line=dict(color='orange', width=4),
                name='Paid'
                )
                line_fig.update_traces(
                selector={'name': 'free'},
                line=dict(color='red', width=4),
                name='Free'
                )

                st.plotly_chart(line_fig)

 
                try:
                # Load model after Spark and data prep
                    tokenizer, model = load_flan_model()

                    prompt_text = engine.build_prompt_from_dataframe(listen_duration)

                    # Generate summary prompt
                    if not listen_duration.empty:
                        prompt_text = engine.build_prompt_from_dataframe(listen_duration)
                        with st.spinner("Generating summary..."):   
                            # Tokenize and generate summary
                            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
                            outputs = model.generate(**inputs, max_length=50, min_length=20, do_sample=True, top_p=0.95, top_k=50)
                            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

                            
                            def capitalize_sentences(text):
                                sentences = text.split('. ')
                                capitalized_sentences = []
                                for sentence in sentences:
                                    words = sentence.split()
                                    capitalized_words = [word.capitalize() if word.islower() else word for word in words]
                                    capitalized_sentence = ' '.join(capitalized_words)
                                    capitalized_sentences.append(capitalized_sentence)
                                return '. '.join(capitalized_sentences)
                                
                            summary = capitalize_sentences(summary)

                            st.write(summary)
                            st.markdown("<p style='font-size: 0.85em; color: gray;'>AI-generated summary</p>", unsafe_allow_html=True)

                    else:
                            st.info("No data available to summarize for selected filters.")
                except Exception as e:
                    st.error(f"Error loading model or generating summary: {e}")
                    st.write("Please check your model and data.")
                
        with col_table[1]:
            with st.container(border=True):
                st.subheader(f"Number of {st.session_state.option} Listens")
                render_map(st.session_state.option)
        
        
                # state_text = st.session_state.location if st.session_state.location != "Nationwide" else "the Nation"
                # st.header(f"Top 10 Artists in {state_text}")
            col_free, col_paid, col_line = st.columns(3)
            with col_paid:
                with st.container(border=True):
                    # paid songs charts
                    paid_text = st.session_state.location if st.session_state.location != "Nationwide" else "the Nation"
                    st.subheader(f'Top Songs for Paid Users in {paid_text}')
                    paid_songs_df = top_paid(_df=clean_listen, state=st.session_state.location)

                    chart_paid_songs = alt.Chart(paid_songs_df).mark_bar().encode(
                        x=alt.X('listens:Q', title='Listens'),
                        y=alt.Y('song:N', sort='-x', title=None),
                        tooltip=['song', 'listens']
                    ).properties(
                        width=700,
                        height=400,
                    ).configure_axis(
                        labelFontSize=14 
                    )
                    st.altair_chart(chart_paid_songs, use_container_width=True)            

            with col_free:
                with st.container(border=True):
                    # free songs chart
                    free_text = st.session_state.location if st.session_state.location != "Nationwide" else "the Nation"
                    st.subheader(f'Top Songs for Free Users in {free_text}')
                    free_songs_df = top_free(_df=clean_listen, state=st.session_state.location)
                    
                    chart_free_songs = alt.Chart(free_songs_df).mark_bar().encode(
                        x=alt.X('listens:Q', title='Listens'),
                        y=alt.Y('song:N', sort='-x', title=None),
                        tooltip=['song', 'listens']
                    ).properties(
                        width=700,
                        height=400,
                    ).configure_axis(
                        labelFontSize=14 
                    )
                    st.altair_chart(chart_free_songs, use_container_width=True)
            
                with col_line:
                    with st.container(border=True):
                        pie_df = create_pie(_df=clean_listen, state=st.session_state.location)
                        pie_state = st.session_state.location if st.session_state.location != "Nationwide" else "the Nation"

                        st.subheader(f"Subscriptions in {pie_state}")

                        # Calculate the percentage column based on 'count' column
                        total = pie_df["count"].sum()  # Calculate total count
                        pie_df["percentage"] = (pie_df["count"] / total) * 100  # Calculate percentage

                        # Create a Pandas DataFrame for the side table
                        percentage_df = pie_df[["subscription", "percentage"]].copy()
                        percentage_df["percentage"] = percentage_df["percentage"].map("{:.1f}%".format)  # Format percentage
                        percentage_df = percentage_df.rename(columns={"subscription": "Subscription", "percentage": "Percentage"})  # Rename columns

                        # Create the pie chart
                        chart = alt.Chart(pie_df).mark_arc(outerRadius=120).encode(
                            theta=alt.Theta(field="count", type="quantitative"),
                            color=alt.Color(field="subscription", type="nominal",
                                            scale=alt.Scale(domain=['free', 'paid'],
                                                            range=['orange', 'blue']),
                                            legend=alt.Legend(orient="bottom")),
                            order=alt.Order(field="count", sort="descending"),
                            tooltip=[
                                "subscription",
                                "count",
                                alt.Tooltip("percentage", format=".1f", title="Percentage (%)")
                            ]
                        )

                        with st.container():

                            # Display chart
                            st.altair_chart(chart, use_container_width=True)

                            # Display styled table below the chart
                            styled_pie_table = percentage_df[['Subscription', 'Percentage']].style.hide(axis="index").set_table_styles([
                                {'selector': 'td', 'props': [('font-size', '20px'), ('text-align', 'left')]},
                                {'selector': 'th', 'props': [('font-size', '20px'), ('text-align', 'left')]}
                            ])
                            st.markdown(styled_pie_table.to_html(), unsafe_allow_html=True)



### ------------------ MAIN UI: TAB 23 ------------------

