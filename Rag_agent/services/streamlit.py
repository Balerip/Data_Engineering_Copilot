import streamlit as st
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'Rag_agent', 'services'))

from chat_service import get_query_response  

st.set_page_config(page_title="Data Engineering Copilot", layout="wide")

st.title("Data Engineering Copilot")
st.markdown("Ask me anything about Spark, dbt, Airflow — or generate a quiz!")


st.sidebar.title("Options")
user_id = st.sidebar.text_input("User ID", value="user_quiz_001")
clear_history = st.sidebar.checkbox("Clear chat history", value=True)


query = st.text_area("Enter your query", height=150, placeholder="e.g., Generate a 5-question quiz on dbt...")

# Submit button
if st.button("Run Copilot"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = get_query_response(query, user_id, clear_history)
                st.markdown("### Copilot Response")
                st.markdown(response.replace("Agent response:", "").strip())
            except Exception as e:
                st.error(f"Error: {e}")
