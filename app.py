import warnings
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # pragma: no cover - compatibility fallback
    InconsistentVersionWarning = Warning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
PRIMARY_MODEL_KEY = "KNN"
FEATURE_COLUMNS = ["duration", "contact", "previous", "housing", "pdays"]
MODEL_FILES = {
    "KNN": {
        "path": ROOT / "knn_model.pkl",
        "label": "KNN",
        "accuracy": 89.09,
        "live_supported": True,
        "comparison_note": "Compared live in the app",
    },
    "MLP": {
        "path": ROOT / "mlp_model.pkl",
        "label": "Neural Network",
        "accuracy": 88.05,
        "live_supported": False,
        "comparison_note": "Offline benchmark only: this model expects a wider 15-feature pipeline than the current demo collects.",
    },
    "Naive_Bayes": {
        "path": ROOT / "naive_bayes_model.pkl",
        "label": "Naive Bayes",
        "accuracy": 85.41,
        "live_supported": True,
        "comparison_note": "Compared live in the app",
    },
}
FEATURE_IMPORTANCE = [
    {
        "feature": "Call Duration",
        "score": 601.82,
        "description": "Duration of the latest customer call. Longer calls usually signal higher intent.",
    },
    {
        "feature": "Previous Contacts",
        "score": 70.06,
        "description": "How often the customer has already been contacted before this campaign.",
    },
    {
        "feature": "Contact Type",
        "score": 57.44,
        "description": "Whether the outreach happened by cellular, telephone, or was unknown.",
    },
    {
        "feature": "Housing Loan",
        "score": 40.32,
        "description": "Whether the customer currently has a housing loan.",
    },
    {
        "feature": "Days Since Previous Contact",
        "score": 37.54,
        "description": "How long it has been since the customer was previously contacted.",
    },
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .hero-card {
            background: linear-gradient(135deg, #0f172a, #1e3a8a);
            color: white;
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            margin-bottom: 1.2rem;
        }
        .hero-card h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.35rem;
        }
        .hero-card p {
            margin: 0;
            color: #dbeafe;
            font-size: 1rem;
        }
        .section-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .result-card {
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
            border: 1px solid transparent;
        }
        .result-card.success {
            background: #ecfdf5;
            border-color: #10b981;
            color: #065f46;
        }
        .result-card.warning {
            background: #fff7ed;
            border-color: #f59e0b;
            color: #9a3412;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.8rem;
            font-weight: 600;
            opacity: 0.85;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    """Load trained models and metadata from disk."""
    try:
        models = {
            key: joblib.load(model_meta["path"])
            for key, model_meta in MODEL_FILES.items()
            if model_meta["live_supported"]
        }
        return models, None
    except FileNotFoundError as exc:
        return None, f"Required model artifact is missing: {exc.filename}"
    except Exception as exc:  # pragma: no cover - defensive Streamlit guard
        return None, str(exc)


def encode_inputs(contact_type: str, has_housing_loan: str) -> tuple[int, int]:
    """Map raw form values into the encoded feature values expected by the models."""
    contact_mapping = {"cellular": 0, "telephone": 1, "unknown": 2}
    housing_mapping = {"no": 0, "yes": 1}
    return contact_mapping[contact_type], housing_mapping[has_housing_loan]


def build_feature_frame(form_values: dict) -> pd.DataFrame:
    contact_encoded, housing_encoded = encode_inputs(
        form_values["contact_type"], form_values["housing_loan"]
    )
    return pd.DataFrame(
        [
            {
                "duration": form_values["duration"],
                "contact": contact_encoded,
                "previous": form_values["previous_contacts"],
                "housing": housing_encoded,
                "pdays": form_values["days_since_contact"],
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def predict_all_models(models: dict, feature_frame: pd.DataFrame) -> tuple[dict, dict]:
    """Run compatible models and capture clear reasons when a model cannot be compared live."""
    results = {}
    unavailable_models = {}
    feature_count = feature_frame.shape[1]

    for key, model in models.items():
        prediction = int(model.predict(feature_frame)[0])
        probabilities = model.predict_proba(feature_frame)[0]
        results[key] = {
            "prediction": prediction,
            "positive_probability": float(probabilities[1]),
            "negative_probability": float(probabilities[0]),
            "confidence": float(max(probabilities)),
            "label": MODEL_FILES[key]["label"],
            "accuracy": MODEL_FILES[key]["accuracy"],
        }

    return results, unavailable_models


def get_recommendation(result: dict) -> tuple[str, str]:
    positive_probability = result["positive_probability"]

    if positive_probability >= 0.7:
        return (
            "High-priority lead",
            "This customer looks worth prioritizing for follow-up because the predicted likelihood is comfortably above the campaign baseline.",
        )
    if positive_probability >= 0.4:
        return (
            "Follow up selectively",
            "This lead is borderline. Consider outreach if the business can support a second contact or wants broader campaign coverage.",
        )
    return (
        "Lower-priority lead",
        "The model suggests this customer is less likely to convert, so marketing effort may be better spent elsewhere first.",
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">Flagship Demo</div>
            <h1>Bank Marketing Prediction System</h1>
            <p>Estimate whether a customer will subscribe to a term deposit and compare how three machine learning models respond to the same lead.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Campaign baseline", "11.7%", "historic subscription rate")
    metric_col2.metric("Best model", "89.09%", "KNN accuracy")
    metric_col3.metric("Dataset size", "4,521", "customer records")


def render_intro() -> None:
    st.markdown(
        """
        <div class="section-card">
            <strong>What this demo shows</strong><br>
            Enter a few customer attributes from the campaign dataset, then review the predicted conversion outcome, confidence, and how alternative models compare. The goal is to make model output understandable to both technical and non-technical stakeholders.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_form() -> dict | None:
    st.subheader("1. Enter customer details")

    with st.form("prediction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            duration = st.slider(
                "Call duration (seconds)",
                min_value=0,
                max_value=3000,
                value=200,
                help="Duration of the most recent campaign call with the customer.",
            )
            contact_type = st.selectbox(
                "Contact communication type",
                options=["cellular", "telephone", "unknown"],
                help="How the customer was contacted.",
            )
            previous_contacts = st.number_input(
                "Previous contacts",
                min_value=0,
                max_value=25,
                value=0,
                help="Number of campaign contacts made before this one.",
            )

        with col2:
            housing_loan = st.selectbox(
                "Housing loan status",
                options=["no", "yes"],
                help="Whether the customer currently has a housing loan.",
            )
            days_since_contact = st.number_input(
                "Days since previous contact",
                min_value=-1,
                max_value=900,
                value=-1,
                help="Use -1 when the customer has never been contacted before.",
            )

        st.caption(
            "The primary decision uses the best-performing KNN model. Model comparison appears after prediction."
        )
        submitted = st.form_submit_button(
            "Evaluate this lead", type="primary", width="stretch"
        )

    if not submitted:
        return None

    return {
        "duration": int(duration),
        "contact_type": contact_type,
        "previous_contacts": int(previous_contacts),
        "housing_loan": housing_loan,
        "days_since_contact": int(days_since_contact),
    }


def create_probability_chart(result: dict) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Will not subscribe", "Will subscribe"],
                y=[result["negative_probability"], result["positive_probability"]],
                marker_color=["#f97316", "#10b981"],
                text=[
                    f"{result['negative_probability']:.1%}",
                    f"{result['positive_probability']:.1%}",
                ],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        height=300,
        yaxis_title="Probability",
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


def create_model_comparison_table(results: dict, unavailable_models: dict) -> pd.DataFrame:
    rows = []

    for key, model_meta in MODEL_FILES.items():
        result = results.get(key)
        if result is None:
            rows.append(
                {
                    "Model": model_meta["label"],
                    "AccuracyScore": model_meta["accuracy"],
                    "Prediction": "Not compared live",
                    "Subscribe probability": "N/A",
                    "Notes": unavailable_models.get(key, model_meta["comparison_note"]),
                }
            )
            continue

        rows.append(
            {
                "Model": result["label"],
                "AccuracyScore": result["accuracy"],
                "Prediction": "Subscribe" if result["prediction"] == 1 else "Not subscribe",
                "Subscribe probability": f"{result['positive_probability']:.1%}",
                "Notes": model_meta["comparison_note"],
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values("AccuracyScore", ascending=False)
    comparison_df["Accuracy"] = comparison_df["AccuracyScore"].map(lambda value: f"{value:.2f}%")
    return comparison_df[
        ["Model", "Accuracy", "Prediction", "Subscribe probability", "Notes"]
    ]


def render_prediction_results() -> None:
    st.subheader("2. Review the prediction")

    if "prediction_results" not in st.session_state:
        st.info("Run the lead through the form above to see the prediction and model comparison.")
        return

    primary_result = st.session_state["prediction_results"][PRIMARY_MODEL_KEY]
    recommendation_title, recommendation_body = get_recommendation(primary_result)
    prediction_label = (
        "Likely to subscribe"
        if primary_result["prediction"] == 1
        else "Unlikely to subscribe"
    )
    card_variant = "success" if primary_result["prediction"] == 1 else "warning"

    left_col, right_col = st.columns([1.15, 1])

    with left_col:
        st.markdown(
            f"""
            <div class="result-card {card_variant}">
                <div class="eyebrow">Primary model: {primary_result['label']}</div>
                <h3 style="margin: 0.3rem 0 0.5rem 0;">{prediction_label}</h3>
                <p style="margin-bottom: 0.6rem;"><strong>{recommendation_title}</strong></p>
                <p style="margin: 0;">{recommendation_body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(
            "Subscription probability",
            f"{primary_result['positive_probability']:.1%}",
        )
        metric_col2.metric(
            "Model confidence",
            f"{primary_result['confidence']:.1%}",
        )

        st.caption(
            "The confidence score reflects how strongly the selected model favors one outcome over the other for this lead."
        )

    with right_col:
        st.plotly_chart(
            create_probability_chart(primary_result),
            width="stretch",
        )


def render_model_comparison() -> None:
    st.subheader("3. Compare the model outputs")

    if "prediction_results" not in st.session_state:
        st.info("Model comparison appears after you run a prediction.")
        return

    comparison_df = create_model_comparison_table(
        st.session_state["prediction_results"],
        st.session_state.get("unavailable_models", {}),
    )
    st.dataframe(comparison_df, width="stretch", hide_index=True)
    st.caption(
        "KNN remains the primary recommendation because it achieved the strongest validation accuracy and the live app includes the exact five-feature path used by the deployed decision flow."
    )


def create_feature_importance_chart() -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                y=[item["feature"] for item in FEATURE_IMPORTANCE],
                x=[item["score"] for item in FEATURE_IMPORTANCE],
                orientation="h",
                marker_color="#60a5fa",
                text=[item["score"] for item in FEATURE_IMPORTANCE],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Top predictive features (ANOVA F-score)",
        xaxis_title="Importance score",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def render_supporting_insights() -> None:
    st.subheader("4. Understand why the model works")

    insight_col1, insight_col2 = st.columns([1.2, 1])

    with insight_col1:
        st.plotly_chart(create_feature_importance_chart(), width="stretch")

    with insight_col2:
        st.markdown(
            """
            <div class="section-card">
                <strong>How to interpret the output</strong><br><br>
                This project does not explain each individual prediction with SHAP or feature attribution. Instead, it shows the strongest global signals discovered during the analysis so recruiters and stakeholders can connect the app back to the modeling work.
            </div>
            """,
            unsafe_allow_html=True,
        )

        for item in FEATURE_IMPORTANCE:
            st.write(f"**{item['feature']}**: {item['description']}")


def render_project_summary() -> None:
    st.markdown("---")
    st.subheader("Project snapshot")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Models compared", "3", "KNN, MLP, Naive Bayes")
    summary_col2.metric("Positive class", "11.7%", "subscription rate")
    summary_col3.metric("Feature set", "5", "selected predictors")

    st.caption(
        "This application packages the end-to-end workflow into a lightweight demo: exploratory analysis, feature selection, model comparison, and a deployment-ready prediction interface."
    )


def main() -> None:
    inject_styles()
    render_header()
    render_intro()

    models, load_error = load_artifacts()

    if load_error:
        st.error("The app could not load its trained model artifacts.")
        st.error(load_error)
        st.info(
            "Check that the .pkl files live beside app.py and that the environment uses a compatible scikit-learn version."
        )
        st.stop()

    form_values = render_input_form()

    if form_values is not None:
        try:
            offline_only_models = {
                key: model_meta["comparison_note"]
                for key, model_meta in MODEL_FILES.items()
                if not model_meta["live_supported"]
            }
            feature_frame = build_feature_frame(form_values)
            prediction_results, unavailable_models = predict_all_models(models, feature_frame)
            unavailable_models.update(offline_only_models)
            st.session_state["prediction_inputs"] = form_values
            st.session_state["prediction_results"] = prediction_results
            st.session_state["unavailable_models"] = unavailable_models
        except Exception as exc:  # pragma: no cover - Streamlit runtime guard
            st.error("The app hit an error while evaluating this lead.")
            st.error(str(exc))

    render_prediction_results()
    render_model_comparison()
    render_supporting_insights()
    render_project_summary()


if __name__ == "__main__":
    main()