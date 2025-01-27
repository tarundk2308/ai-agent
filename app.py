from langchain_experimental.agents.agent_toolkits import create_csv_agent
import streamlit as st
import os
import tempfile
import chardet  # For encoding detection
from langchain_groq import ChatGroq

# Initialize the LLM model
model = ChatGroq(model="llama-3.1-8b-instant", api_key='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')

def detect_encoding(file):
    """Detect the encoding of a file using chardet."""
    result = chardet.detect(file.read())
    file.seek(0)  # Reset file pointer to the beginning
    return result['encoding']

def main():
    # Configure Streamlit page
    st.set_page_config(page_title="Ask Your Medical Data", page_icon="🩺")
    st.header("🩺 Medical Data Question-Answering App")
    st.subheader("Upload your medical CSV data and get answers to your queries!")

    # Allow the user to upload a CSV file
    file = st.file_uploader("Upload your medical data (CSV format)", type="csv")

    if file is not None:
        try:
            # Detect the file encoding
            encoding = detect_encoding(file)

            # Create a temporary file to store the uploaded CSV data
            with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv", delete=False) as f:
                # Convert bytes to a string before writing to the file
                data_str = file.getvalue().decode(encoding)
                f.write(data_str)
                f.flush()

                # Ask the user to input a question
                st.markdown("### Ask a question about your medical data:")
                user_input = st.text_input("Enter your question:")

                if user_input:
                    # Add a custom prompt for the model
                    prompt = f"""
                    You are a highly intelligent medical assistant. You are given a medical dataset in CSV format. 
                    Answer questions based on the dataset, such as aggregations, counts, averages, and any specific medical insights. 
                    Ensure that the responses are clear and concise. 

                    User Question: {user_input}
                    """

                    # Create a CSV agent using the specified language model and temporary file
                    agent = create_csv_agent(
                        model,
                        f.name,
                        verbose=True,allow_dangerous_code=True,
                        agent_executor_kwargs={"handle_parsing_error": True}
                    )

                    # Run the agent on the user's question with the custom prompt
                    st.markdown("### Response:")
                    response = agent.run(prompt)
                    st.write(response)

        except UnicodeDecodeError as e:
            st.error(f"Error decoding the file: {e}. Please upload a valid CSV file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
