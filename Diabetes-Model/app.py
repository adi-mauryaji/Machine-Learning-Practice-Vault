from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

INK = "#1E2A2C"
MUTED = "#596462"
GRID = "#EAE5D8"
LIKELY = "Towards likely"
UNLIKELY = "Towards unlikely"
LIKELY_COLOR = "#BF4A28"
UNLIKELY_COLOR = "#00806B"

folder = Path(__file__).parent

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")
st.html(folder / "style.css")


# Load the saved files once, not on every slider move
@st.cache_resource
def load_files():
    return joblib.load(folder / "model.pkl"), joblib.load(folder / "scalar.pkl")


def section(number, title):
    st.html(f'<p class="section"><span>{number}</span>{title}</p>')


def signed(number):
    return f"{number:+.2f}".replace("-", "−")


model, scaler = load_files()

# Header
st.html('<p class="eyebrow">Practice ML project</p>')
st.title("Diabetes Prediction App")
st.html("""
<div class="specs">
  <div><span>Model</span>Linear SVM</div>
  <div><span>Data</span>768 records</div>
  <div><span>Test accuracy</span>70.8%</div>
  <div class="disclaimer">For my ML practice, not medical advice.</div>
</div>
""")

inputs, report = st.columns([1.05, 1], gap="large")

# Label, minimum, maximum, starting value.
# The ranges come from diabetes.csv and the starting values are its first record.
with inputs:
    st.html('<p class="note intro">Starting values are the first record in the data. Move any slider and the report updates straight away.</p>')

    section("01", "Patient")
    left, right = st.columns(2)
    age = left.slider("Age", 21, 81, 50)
    pregnancies = right.slider("Pregnancies", 0, 17, 6)

    section("02", "Blood tests")
    left, right = st.columns(2)
    glucose = left.slider("Glucose (mg/dL)", 44, 199, 148, help="Two hours into a glucose tolerance test.")
    insulin = right.slider("Insulin (μU/mL)", 0, 846, 0, help="Two hours into the same test. 0 means not measured.")

    section("03", "Body and family history")
    left, right = st.columns(2)
    bmi = left.slider("BMI (kg/m²)", 18.2, 67.1, 33.6, 0.1, format="%.1f")
    blood_pressure = right.slider("Blood pressure (mm Hg)", 24, 122, 72, help="Diastolic, the lower number.")
    skin_thickness = left.slider("Skin thickness (mm)", 0, 99, 35, help="Triceps skin fold. 0 means not measured.")
    pedigree = right.slider(
        "Diabetes pedigree function", 0.078, 2.42, 0.627, 0.001,
        format="%.3f", help="A score based on diabetes in the family.",
    )

    section("04", "About this data")
    st.html("""
    <p class="note">
      Test accuracy is 70.8%, so about 3 in 10 test predictions were wrong.
      The data is 768 records of women aged 21 and over (Pima Indians Diabetes Database),
      and the sliders cover the ranges seen in it.
      <a href="https://github.com/adi-mauryaji/Machine-Learning-Practice-Vault">Source code</a>
    </p>
    """)

data = pd.DataFrame([{
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": pedigree,
    "Age": age,
}])

scaled_data = scaler.transform(data)
prediction = model.predict(scaled_data)
score = model.decision_function(scaled_data)[0]

# In a linear SVM: score = baseline + (weight x scaled value) for each input
baseline = model.intercept_[0]
effects = model.coef_[0] * scaled_data[0]

if prediction[0] == 1:
    verdict_style, stamp, verdict = "likely", "▲ Likely", "The person is likely to have diabetes."
else:
    verdict_style, stamp, verdict = "unlikely", "▼ Unlikely", "The person is unlikely to have diabetes."

# The meter runs from -4 to +4, which covers almost every record in the data
marker = min(max((score + 4) / 8 * 100, 0), 100)

# Keep zero in the middle, and widen the chart only when a bar needs more room
chart_limit = max(1.5, abs(effects).max())

chart_data = pd.DataFrame({
    "Input": ["Pregnancies", "Glucose", "Blood pressure", "Skin thickness", "Insulin", "BMI", "Pedigree", "Age"],
    "Value": [f"{value:g}" for value in data.iloc[0]],
    "Data average": scaler.mean_.round(2),
    "Effect on score": effects.round(2),
    "Direction": [LIKELY if effect > 0 else UNLIKELY for effect in effects],
    "Size": abs(effects),
})

base = alt.Chart(chart_data).encode(
    x=alt.X(
        "Effect on score:Q",
        title="Effect on model score",
        scale=alt.Scale(domain=[-chart_limit, chart_limit], nice=False),
        axis=alt.Axis(
            values=[tick for tick in range(-3, 4) if abs(tick) <= chart_limit],
            format="d",
            labelExpr="datum.value > 0 ? '+' + datum.label : datum.label",
        ),
    ),
    y=alt.Y(
        "Input:N",
        title=None,
        sort=chart_data.sort_values("Size", ascending=False)["Input"].tolist(),
        axis=alt.Axis(minExtent=100),
    ),
    color=alt.Color(
        "Direction:N",
        scale=alt.Scale(domain=[UNLIKELY, LIKELY], range=[UNLIKELY_COLOR, LIKELY_COLOR]),
        legend=alt.Legend(orient="top", title=None, symbolType="square"),
    ),
    tooltip=["Input", "Value", "Effect on score"],
)

# Round only the far end of each bar, so the bars start flat on the zero line
bars_right = base.transform_filter(alt.datum.Direction == LIKELY).mark_bar(
    size=14, cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
)
bars_left = base.transform_filter(alt.datum.Direction == UNLIKELY).mark_bar(
    size=14, cornerRadiusTopLeft=4, cornerRadiusBottomLeft=4,
)
zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK).encode(x=alt.X("x:Q", title="Effect on model score"))

chart = (
    alt.layer(bars_left, bars_right, zero_line)
    .properties(height=250, background="transparent")
    .configure_view(stroke=None)
    .configure_axis(
        labelFont="IBM Plex Sans", titleFont="IBM Plex Sans", labelFontSize=12, titleFontSize=12,
        labelColor=INK, titleColor=MUTED, titleFontWeight=400,
        gridColor=GRID, domain=False, ticks=False,
    )
    .configure_legend(labelFont="IBM Plex Sans", labelFontSize=12, labelColor=INK)
)

with report:
    with st.container(key="report"):
        st.html(f"""
        <div class="verdict {verdict_style}">
          <span class="stamp">{stamp}</span>
          <p>{verdict}</p>
        </div>
        <div class="score">
          <div class="score-head"><span>Model score</span><strong>{signed(score)}</strong></div>
          <div class="track"><i style="left: {marker:.1f}%"></i></div>
          <div class="track-labels"><div>◀ Unlikely</div><div>0</div><div>Likely ▶</div></div>
          <p class="note">Above 0 means likely. Closer to 0 means a closer call. This is not a probability.</p>
        </div>
        """)

        st.html("""
        <p class="report-heading">What pushed this result</p>
        <p class="note">What this model learned from its data, not medical causes.</p>
        """)
        st.altair_chart(chart, theme=None, width="stretch")

        with st.expander("See the numbers"):
            st.dataframe(chart_data[["Input", "Value", "Data average", "Effect on score"]], hide_index=True)
            st.caption(f"Baseline {signed(baseline)}, plus effects {signed(effects.sum())}, gives score {signed(score)}")
