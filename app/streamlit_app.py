
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "risk_model.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"



st.set_page_config(
    page_title="AI Risk Analyzer",
    layout="wide",
)


# --------------------------------------------------
# Load trained model and metadata
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"Loading model from: {MODEL_PATH}")
    print(f"Model size: {MODEL_PATH.stat().st_size} bytes")

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not metadata.get("features"):
        raise ValueError(
            "Metadata must contain a non-empty 'features' list."
        )

    return metadata


def validate_feature_schema(model, metadata):
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None:
        return

    model_features = list(model_features)
    metadata_features = [feature["name"] for feature in metadata["features"]]
    if model_features == metadata_features:
        return

    missing = [name for name in model_features if name not in metadata_features]
    unexpected = [name for name in metadata_features if name not in model_features]
    raise ValueError(
        "Model and metadata feature definitions do not match. "
        f"Missing from metadata: {missing}; unexpected in metadata: "
        f"{unexpected}; feature order must also match."
    )


# --------------------------------------------------
# Interface
# --------------------------------------------------

st.title("AI Risk Analyzer")
st.subheader("Machine Learning for Financial Risk Assessment")

st.write(
    "Enter financial indicators to estimate risk using a "
    "trained machine learning model."
)

st.info(
    "This tool provides a model-based estimate, not a guarantee "
    "of future financial performance or a substitute for "
    "professional financial analysis."
)

try:
    model = load_model()
    metadata = load_metadata()
    validate_feature_schema(model, metadata)
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Application is not ready: {error}")
    st.markdown(
        """
        **To enable predictions:**
        1. Run `python -m src.train_risk_model` from the project directory.
        2. Restart the application after training completes.
        """
    )
    st.stop()


# --------------------------------------------------
# Input form
# --------------------------------------------------

features = metadata["features"]
positive_class = metadata.get("positive_class")
default_threshold = float(metadata.get("threshold", 0.5))
default_threshold = min(max(default_threshold, 0.0), 1.0)

st.header("Company Financial Indicators")

st.caption(
    "Input fields must match the features used during model training."
)

input_values = {}

with st.form("risk_assessment_form"):
    columns = st.columns(2)

    for index, feature in enumerate(features):
        name = feature["name"]
        label = feature.get("label", name)
        feature_type = feature.get("type", "number")

        with columns[index % 2]:
            if feature_type == "categorical":
                options = feature.get("options", [])

                if not options:
                    st.warning(
                        f"No options configured for {label}. "
                        "Check model_metadata.json."
                    )
                    continue

                default_value = feature.get("default", options[0])

                if default_value not in options:
                    default_value = options[0]

                input_values[name] = st.selectbox(
                    label,
                    options=options,
                    index=options.index(default_value),
                    help=feature.get("help"),
                )

            else:
                minimum = float(feature.get("min", 0.0))
                maximum = float(feature.get("max", 100.0))
                default = float(
                    feature.get("default", (minimum + maximum) / 2)
                )
                step = float(feature.get("step", 0.1))

                if minimum > maximum:
                    st.error(f"Invalid range configured for {label}.")
                    continue

                default = min(max(default, minimum), maximum)

                input_values[name] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    help=feature.get("help"),
                )

    threshold = st.slider(
        "Risk decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=min(max(default_threshold, 0.05), 0.95),
        step=0.05,
        help=(
            "A case is flagged as high risk when its predicted "
            "positive-class probability reaches this threshold. "
            "The threshold should be selected using validation data."
        ),
    )

    submitted = st.form_submit_button(
        "Assess Financial Risk",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:
    expected_features = [feature["name"] for feature in features]

    if len(input_values) != len(expected_features):
        st.error(
            "Some input fields are missing. Check the feature "
            "definitions in model_metadata.json."
        )
        st.stop()

    input_df = pd.DataFrame(
        [input_values],
        columns=expected_features,
    )

    try:
        prediction = model.predict(input_df)[0]

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if positive_class is not None:
                matching_classes = [
                    index
                    for index, class_value in enumerate(classes)
                    if str(class_value) == str(positive_class)
                ]

                if not matching_classes:
                    st.error(
                        "The configured positive_class does not match "
                        "the classes learned by the model."
                    )
                    st.stop()

                positive_index = matching_classes[0]
            else:
                positive_index = len(classes) - 1

            probability = float(probabilities[positive_index])

        st.divider()
        st.header("Assessment Results")

        if probability is not None:
            risk_flag = probability >= threshold

            first_col, second_col = st.columns(2)

            with first_col:
                st.metric(
                    "Predicted probability of the target event",
                    f"{probability:.1%}",
                )

            with second_col:
                st.metric(
                    "Decision threshold",
                    f"{threshold:.0%}",
                )

            if risk_flag:
                st.error(
                    "The case is flagged by the selected threshold."
                )
            else:
                st.success(
                    "The case is below the selected decision threshold."
                )

            st.progress(probability)

            st.caption(
                "The probability refers to the configured positive class. "
                "It should not be interpreted as a reliable real-world "
                "risk probability unless calibration has been evaluated."
            )

        else:
            st.write("Predicted class:", str(prediction))
            st.warning(
                "This model does not expose predict_proba(). "
                "A probability-based risk estimate is unavailable."
            )

        with st.expander("View submitted indicators"):
            st.dataframe(input_df, use_container_width=True)

        with st.expander("Interpretation and limitations"):
            st.write(
                "This result reflects patterns learned from the training "
                "dataset. It does not establish causation and may be "
                "unreliable for companies or cases that differ from the "
                "training data."
            )
            st.write(
                "The selected threshold changes the classification "
                "decision, not the model's predicted probability."
            )

    except (ValueError, TypeError, KeyError) as error:
        st.error(
            f"Prediction failed. Check the input types and feature "
            f"definitions: {error}"
        )
    except Exception as error:
        st.error(
            f"An unexpected prediction error occurred: {error}"
        )
