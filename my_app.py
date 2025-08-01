# Read labels_and_hours as a dictionary from final_labels_and_hours.csv
import pandas as pd
import chromadb
import streamlit as st
import os
from main import get_closest_label_from_chroma_db, preprocess_ticket
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

def get_visualization_path(label):
    """
    Get the path to the visualization file for a given label.
    
    Args:
        label (str): The label to find visualization for
    
    Returns:
        str or None: Path to the visualization file if it exists, None otherwise
    """
    # List all files in the visualizations directory
    viz_dir = "visualizations"
    if not os.path.exists(viz_dir):
        return None
    
    # Get all PNG files in the visualizations directory
    viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
    
    # Try to find a matching file
    for viz_file in viz_files:
        # Remove the prefix and suffix to get the label part
        if viz_file.startswith('hours_distribution_') and viz_file.endswith('.png'):
            file_label = viz_file[len('hours_distribution_'):-len('.png')]
            
            # Compare labels with some flexibility for special characters
            if file_label == label or file_label.replace('_', '/') == label or file_label.replace('/', '_') == label:
                return os.path.join(viz_dir, viz_file)
    
    # If exact match not found, try partial matching
    for viz_file in viz_files:
        if viz_file.startswith('hours_distribution_') and viz_file.endswith('.png'):
            file_label = viz_file[len('hours_distribution_'):-len('.png')]
            # Check if the core part of the label matches (ignoring special characters)
            label_clean = label.replace('/', '').replace('_', '').replace(' ', '').lower()
            file_label_clean = file_label.replace('/', '').replace('_', '').replace(' ', '').lower()
            
            if label_clean == file_label_clean:
                return os.path.join(viz_dir, viz_file)
    
    return None

def run_pipeline(short_description, description):
    """
    Run the pipeline with user-provided descriptions.
    
    Args:
        short_description (str): Short description of the ticket
        description (str): Detailed description of the ticket
    
    Returns:
        tuple: (closest_label, predicted_hours)
    """
    # Use labels_and_hours.csv instead of final_labels_and_hours.csv
    labels_and_hours_df = pd.read_csv('final_labels_for_demo.csv')
    labels_and_hours = {}
    
    for index, row in labels_and_hours_df.iterrows():
        if pd.notna(row['Hours']):  # Check if Hours is not NaN
            try:
                labels_and_hours[row['Label']] = eval(row['Hours'])
            except:
                # Skip rows with invalid data
                continue

    chroma_client = chromadb.CloudClient(
        api_key=os.getenv('CHROMA_API_KEY'),
        tenant=os.getenv('CHROMA_TENANT'),
        database=os.getenv('CHROMA_DATABASE')
    )
    
    # Preprocess the ticket descriptions
    processed_short_desc, processed_desc = preprocess_ticket(short_description, description)
    
    # Combine descriptions for better matching
    combined_description = f"{processed_short_desc} {processed_desc}"
    
    closest_label, closest_definition = get_closest_label_from_chroma_db(combined_description, chroma_client)
    if closest_label in labels_and_hours:
        # Predict hours based on training data
        predicted_hours = sum(labels_and_hours[closest_label]) / len(labels_and_hours[closest_label])
        predicted_hours = round(predicted_hours, 2)
    else:
        predicted_hours = "Could not predict hours, label not found in train set"
    return closest_label, predicted_hours

def main():
    """
    Streamlit main function for the Ticket Effort Estimator UI.
    """
    st.set_page_config(
        page_title="Ticket Effort Estimator",
        page_icon="🎫",
        layout="wide"
    )
    
    st.title("🎫 Ticket Effort Estimator")
    st.markdown("---")
    
    st.markdown("### Enter Ticket Information")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        short_description = st.text_input(
            "Short Description",
            placeholder="Enter a brief summary of the issue...",
            help="Provide a concise summary of the ticket"
        )
    
    with col2:
        description = st.text_area(
            "Detailed Description",
            placeholder="Enter detailed description of the issue...",
            height=100,
            help="Provide detailed information about the issue"
        )
    
    # Estimate button
    if st.button("🔍 Estimate Effort", type="primary", use_container_width=True):
        if short_description.strip() and description.strip():
            with st.spinner("Analyzing ticket and predicting effort..."):
                try:
                    closest_label, predicted_hours = run_pipeline(short_description, description)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Results")
                    
                    # Create two columns for results
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.markdown("**🏷️ Closest Category**")
                        st.markdown(f"<div style='font-size: 1.25rem; font-weight: 600; color: rgb(49, 51, 63); word-wrap: break-word;'>{closest_label}</div>", unsafe_allow_html=True)
                    
                    with result_col2:
                        if isinstance(predicted_hours, str):
                            st.error(predicted_hours)
                        else:
                            st.metric(
                                label="⏱️ Predicted Hours",
                                value=f"{predicted_hours} hours"
                            )
                    
                    # Additional information
                    st.markdown("---")
                    st.markdown("### ℹ️ Information")
                    st.info(
                        f"Based on historical data, tickets similar to '{closest_label}' "
                        f"typically require approximately {predicted_hours} hours to resolve."
                    )
                    
                    # Display visualization if available
                    viz_path = get_visualization_path(closest_label)
                    if viz_path:
                        st.markdown("---")
                        st.markdown("### 📈 Hours Distribution Visualization")
                        st.image(viz_path, caption=f"Hours distribution for {closest_label} from historical data", use_container_width=True)
                    else:
                        st.markdown("---")
                        st.markdown("### 📈 Visualization")
                        st.info("📊 No visualization available for this category.")
                    
                except Exception as e:
                    st.error(f"An error occurred while processing your request: {str(e)}")
                    st.markdown("Please check your input and try again.")
        else:
            st.warning("⚠️ Please fill in both the short description and detailed description fields.")
    
    # Add some helpful information
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **Ticket Effort Estimator** uses machine learning to predict the effort required to resolve support tickets:
        
        1. **Input Processing**: Your ticket descriptions are preprocessed and cleaned
        2. **Similarity Matching**: The system finds the most similar ticket category from historical data
        3. **Effort Prediction**: Based on past tickets in that category, it estimates the required effort
        
        **Tips for better predictions:**
        - Provide clear, detailed descriptions
        - Include specific technical terms or error messages
        - Mention the systems or components involved
        """)

if __name__ == "__main__":
    main()