import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import pickle

st.title("Crash Injury/Fatality Predictor")
rf_model = pickle.load(open('r_forest_EMT.pkl', 'rb'))


# Sidebar Formatting
author_pic = Image.open("./pictures/stephen.jpg")
st.sidebar.image(author_pic, "Your humble app creator", use_column_width=True)
st.sidebar.markdown("[Hello](https://github.com/S-DeFerrari)")
st.sidebar.write("This app is powered by Machine Learning!")
st.sidebar.write("It uses a Random Forest Classification model "
                 "trained with car crash information obtained from NY State's DMV in order to predict whether an "
                 "accident involves an injury/fatality or if it's just property damage. This model was originally "
                 "envisioned as a way to assist 911 operators in assigning ambulance services.")
st.sidebar.write("With a .35 threshold it was accurate 68% of the time, had a precision score of 58%, a recall score "
                 "of 72%, and an F beta score of 69% with a 2 beta")
threshold = st.sidebar.slider("Adjust the classification threshold to suit your needs:", .20, .80, .35, .15)

rf_20 = Image.open("./pictures/random_forest_matrix_20.jpg")
if threshold == 20:
    st.sidebar.image(rf_20, use_column_width=True)
rf_35 = Image.open("./pictures/random_forest_matrix_35.jpg")
if threshold == 35:
    st.sidebar.image(rf_35, use_column_width=True)
rf_50 = Image.open("./pictures/random_forest_matrix_50.jpg")
if threshold == 50:
    st.sidebar.image(rf_50, use_column_width=True)
rf_65 = Image.open("./pictures/random_forest_matrix_65.jpg")
if threshold == 65:
    st.sidebar.image(rf_65, use_column_width=True)
rf_80 = Image.open("./pictures/random_forest_matrix_80.jpg")
if threshold == 80:
    st.sidebar.image(rf_80, use_column_width=True)

# Main column Formatting
ambulance_pic = Image.open("./pictures/ambulance.png")
st.image(ambulance_pic, use_column_width=True)
st.write("A 2014 study by New York State listed motor vehicle traffic injuries as the 4th leading cause of injury "
         "related deaths. EMTs arriving to a scene fast can be the difference between life and death, but how can "
         "911 operators ensure they're sending ambulances to the right accidents and not wasting resources? Fill "
         "out the following questions and hit 'generate prediction' to have machine learning assist you in your "
         "decision.")

# Time and Location questions
st.subheader("Time and Location")
counties = {'Bronx': 32900.43, 'Kings': 35367.13, 'New York': 69464.43, 'Queens': 20553.97,
            'Richmond': 8030.32, 'Albany': 581.87, 'Allegany': 47.55, 'Broome': 284.23, 'Cattaraugus': 61.39,
            'Cayuga': 115.71, 'Chautauqua': 127.24, 'Chemung': 218.07, 'Chenango': 56.49, 'Clinton': 79.13,
            'Columbia': 99.41, 'Cortland': 98.92, 'Delaware': 33.26, 'Dutchess': 373.9, 'Erie': 881.41,
            'Essex': 21.94, 'Franklin': 31.67, 'Fulton': 112.08, 'Genesee': 121.88, 'Greene': 76.06, 'Hamilton': 2.82,
            'Herkimer': 45.71, 'Jefferson': 91.62, 'Lewis': 21.25, 'Livingston': 103.51, 'Madison': 112.15,
            'Monroe': 1132.58, 'Montgomery': 124.6, 'Nassau': 4704.73, 'Niagara': 414.41, 'Oneida': 193.72,
            'Onondaga': 599.99, 'Ontario': 167.58, 'Orange': 459.3, 'Orleans': 109.6, 'Oswego': 128.31,
            'Otsego': 62.15, 'Putnam': 432.94, 'Rensselaer': 244.36, 'Rockland': 1795.95, 'St Lawrence': 41.76,
            'Saratoga': 271.13, 'Schenectady': 756.54, 'Schoharie': 52.67, 'Schuyler': 55.87, 'Seneca': 108.9,
            'Steuben': 71.19, 'Suffolk': 1637.36, 'Sullivan': 80.1, 'Tioga': 98.58, 'Tompkins': 213.98,
            'Ulster': 162.33, 'Warren': 75.79, 'Washington': 76.06, 'Wayne': 155.3, 'Westchester': 2204.68,
            'Wyoming': 71.12,'Yates': 74.96}
county_choices = list(counties.keys())
county_st = st.selectbox("Which County are you in?", options=county_choices)
county_st = counties[county_st]
month_st = st.slider("What Month is it (1 = January, 12 = December)", 1, 12, 1)
day_st = st.slider("What day of the week is it? (1 = Monday, 7 = Sunday)", 1, 7, 1)
hour_st = st.slider("What hour is it? (24 hour clock)", 0, 23, 1)

# Visibility
st.subheader("Visibility")
twilight_st = st.checkbox("It is the Twilight Hour (dusk or dawn)")
daylight_st = st.checkbox("It is currently daylight")
lit_dark_st_st = st.checkbox("If the accident occurred at night, it was on a lit dark road")
precipitation_st = st.checkbox("It is currently raining, snowing, sleeting, etc")

# Road Conditions
st.subheader("Road Conditions")
dry_road_st = st.checkbox("Road conditions are dry")
straight_st = st.checkbox("The accident occurred on a straight away")
level_st = st.checkbox("The accident occurred on a a level road")
crest_st = st.checkbox("The accident occurred at the crest (top area) of a hill")

# Crash info
st.subheader("Crash Information")
police_st = st.checkbox("A police report is being made already")
traffic_control_st = st.checkbox("A traffic control device (stop sign, traffic light) is nearby")
other_vic_st = st.checkbox("Another vehicle was involved")
vics_st = st.number_input("How many vehicles are involved?", value=0, step=1)
collision_st = st.checkbox("There was a collision of some sort")
pedestrian_st = st.checkbox("A pedestrian was involved")
cyclist_st = st.checkbox("A cyclist was involved")
fixed_st = st.checkbox("A fixed object was involved")
fire_st = st.checkbox("There was an explosion or fire")
animal_st = st.checkbox("An animal was involved")
overturn_st = st.checkbox("A vehicle was overturned")
train_st = st.checkbox("A train was involved")
submerged_st = st.checkbox("A vehicle was submerged")
unknown_st = st.checkbox("The cause of the accident is unknown")

st.subheader("Injury/Fatality Prediction")
# Final
if st.button("Generate Prediction"):
    fate_list = [police_st, traffic_control_st, vics_st, county_st, hour_st, month_st, day_st, precipitation_st,
                 daylight_st, lit_dark_st_st, twilight_st, straight_st, level_st, crest_st, dry_road_st, collision_st,
                 pedestrian_st, other_vic_st, cyclist_st, fixed_st, unknown_st, fire_st, animal_st, overturn_st,
                 train_st, submerged_st]

    prediction = (rf_model.predict_proba([fate_list])[:,1] >= threshold ).astype(bool)

    if prediction[0] == 0:
        st.markdown("🅾️")
        st.write(f"This model does not predict that an injury/fatality is involved.")

    if prediction[0] == 1:
       st.markdown("✅")
       st.write(f"This model predicts that there is an injury/fatality involved, it advises "
                f"sending an ambulance to this crash.")


    st.write("Presented below is the confusion matrix for this model at your chosen threshold on validation data. "
             "Consider the variation between false-negatives and true-negatives along with false positives and "
             "true positives when taking its advice.")
    rf_20 = Image.open("./pictures/random_forest_matrix_20.jpg")
    if threshold == .20:
        st.image(rf_20, use_column_width=True)
    rf_35 = Image.open("./pictures/random_forest_matrix_35.jpg")
    if threshold == .35:
        st.image(rf_35, use_column_width=True)
    rf_50 = Image.open("./pictures/random_forest_matrix_50.jpg")
    if threshold == .50:
        st.image(rf_50, use_column_width=True)
    rf_65 = Image.open("./pictures/random_forest_matrix_65.jpg")
    if threshold == .65:
        st.image(rf_65, use_column_width=True)
    rf_80 = Image.open("./pictures/random_forest_matrix_80.jpg")
    if threshold == .80:
        st.image(rf_80, use_column_width=True)
