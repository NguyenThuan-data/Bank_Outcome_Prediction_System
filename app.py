import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-success {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 0.25rem;
    margin: 1rem 0;
}
.prediction-warning {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
    padding: 1rem;
    border-radius: 0.25rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the pre-trained models and encoders"""
    try:
        # Load models
        knn_model = joblib.load('knn_model.pkl')
        mlp_model = joblib.load('mlp_model.pkl')
        nb_model = joblib.load('naive_bayes_model.pkl')
        model_info = joblib.load('model_info.pkl')
        
        models = {
            'KNN (Best - 89.09%)': knn_model,
            'Neural Network (88.05%)': mlp_model,
            'Naive Bayes (85.41%)': nb_model
        }
        
        return models, model_info
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.error("Please make sure the .pkl files are in the same directory!")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Bank Marketing Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("### Predict whether a customer will subscribe to a term deposit")
    
    # Load models
    models, model_info = load_models()
    
    if models is None:
        st.stop()
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Customer Information")
        
        # Input form
        with st.form("prediction_form"):
            # Duration (most important feature)
            duration = st.slider(
                "📞 Call Duration (seconds)", 
                min_value=0, max_value=3000, value=200,
                help="Duration of the last contact call with the customer"
            )
            
            # Contact type
            contact_type = st.selectbox(
                "📱 Contact Communication Type",
                options=['cellular', 'telephone', 'unknown'],
                index=0,
                help="How was the customer contacted?"
            )
            
            # Previous contacts
            previous_contacts = st.number_input(
                "📊 Previous Contacts", 
                min_value=0, max_value=25, value=0,
                help="Number of contacts performed before this campaign"
            )
            
            # Housing loan
            has_housing_loan = st.selectbox(
                "🏠 Housing Loan Status",
                options=['no', 'yes'],
                help="Does the customer have a housing loan?"
            )
            
            # Days since previous contact
            days_since_contact = st.number_input(
                "📅 Days Since Previous Contact",
                min_value=-1, max_value=900, value=-1,
                help="Days passed since last contact (-1 = never contacted before)"
            )
            
            # Model selection
            selected_model = st.selectbox(
                "🤖 Choose Prediction Model",
                options=list(models.keys()),
                help="KNN model performed best in the analysis"
            )
            
            # Submit button
            submitted = st.form_submit_button("🔮 Make Prediction", type="primary")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if submitted:
            # Encode categorical inputs with error handling
            try:
                # Manual encoding based on your original analysis
                # From your notebook, the encoding should be:
                contact_mapping = {'cellular': 0, 'telephone': 1, 'unknown': 2}
                housing_mapping = {'no': 0, 'yes': 1}
                
                # Encode using manual mapping (more reliable)
                contact_encoded = contact_mapping.get(contact_type, 0)  # default to 0 if not found
                housing_encoded = housing_mapping.get(has_housing_loan, 0)  # default to 0 if not found
                
                # Alternative: Try using saved encoders with fallback
                # try:
                #     contact_encoded = model_info['label_encoders']['contact'].transform([contact_type])[0]
                #     housing_encoded = model_info['label_encoders']['housing'].transform([has_housing_loan])[0]
                # except (KeyError, ValueError):
                #     # Fallback to manual encoding
                #     contact_encoded = contact_mapping.get(contact_type, 0)
                #     housing_encoded = housing_mapping.get(has_housing_loan, 0)
                
                # Prepare input data (order: duration, contact, previous, housing, pdays)
                input_features = np.array([[duration, contact_encoded, previous_contacts, housing_encoded, days_since_contact]])
                
                # Make prediction
                model = models[selected_model]
                prediction = model.predict(input_features)[0]
                probabilities = model.predict_proba(input_features)[0]
                
                # Display prediction result
                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-success">
                        <h3>✅ Customer WILL Subscribe!</h3>
                        <p>This customer is likely to subscribe to the term deposit.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    confidence = probabilities[1]
                else:
                    st.markdown("""
                    <div class="prediction-warning">
                        <h3>❌ Customer will NOT Subscribe</h3>
                        <p>This customer is unlikely to subscribe to the term deposit.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    confidence = probabilities[0]
                
                # Confidence metric
                st.metric("🎯 Confidence Level", f"{confidence:.1%}")
                
                # Probability chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Will NOT Subscribe', 'WILL Subscribe'],
                        y=[probabilities[0], probabilities[1]],
                        marker_color=['#ff7f7f', '#7fbf7f'],
                        text=[f'{probabilities[0]:.1%}', f'{probabilities[1]:.1%}'],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Prediction Probability",
                    yaxis_title="Probability",
                    showlegend=False,
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        
        else:
            st.info("👆 Fill in the customer information and click 'Make Prediction' to get results!")
    
    # Model information section
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric(
            label="🥇 KNN Model",
            value="89.09%",
            delta="Best Performance"
        )
    
    with perf_col2:
        st.metric(
            label="🥈 Neural Network",
            value="88.05%", 
            delta="Good Performance"
        )
    
    with perf_col3:
        st.metric(
            label="🥉 Naive Bayes",
            value="85.41%",
            delta="Baseline Model"
        )
    
    # Feature importance section
    with st.expander("📈 See Feature Importance"):
        st.write("**Top 5 Features from Analysis:**")
        
        importance_data = {
            'Feature': ['Call Duration', 'Previous Contacts', 'Contact Type', 'Housing Loan', 'Days Since Contact'],
            'Importance Score': [601.82, 70.06, 57.44, 40.32, 37.54],
            'Description': [
                'Duration of the last contact call',
                'Number of contacts before this campaign',
                'Communication type (cellular/telephone)',
                'Whether customer has housing loan',
                'Days since previous contact'
            ]
        }
        
        # Create importance chart
        fig_importance = go.Figure(data=[
            go.Bar(
                y=importance_data['Feature'],
                x=importance_data['Importance Score'],
                orientation='h',
                marker_color='lightblue',
                text=importance_data['Importance Score'],
                textposition='auto',
            )
        ])
        
        fig_importance.update_layout(
            title="Feature Importance (ANOVA F-Score)",
            xaxis_title="Importance Score",
            height=400
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Feature descriptions
        for i, feature in enumerate(importance_data['Feature']):
            st.write(f"**{feature}:** {importance_data['Description'][i]}")

if __name__ == "__main__":
    main()