#importing Libraries

import streamlit as st
import tensorflow as tf
import numpy as np
import time

#model prediction

def model_prediction(test_image):
    model = tf.keras.models.load_model( 'trained_plant_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size = (128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction =  model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


#sidebar

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("", ["Home", "About","Disease Recognition"])


#Home page

if(app_mode == "Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "image.png"
    st.image(image_path, use_column_width=True)
    st.markdown(""" ### Welcome to the **Plant Disease Recognition System**

Our mission is to assist in identifying plant diseases quickly and accurately. Simply upload an image, and our system will automatically detect and diagnose the disease.

### How It Works:

- **Upload Image** – Submit a photo of the plant.
- **Analysis** – The system processes the image to identify potential diseases.
- **Result** – Instantly receive the disease name.

### Why Choose Us:

- High Accuracy
- User-Friendly Experience
- Time-Saving and Efficient
                """)
    
#about page

elif(app_mode=="About"):
    st.header("Header")
    st.markdown("""
    ### About Dataset
    Original Dataset is available here: 
    The dataset is recreated using offline augmentation from the original dataset.
    
    ## Content
    1. Train (70295 images)
    2. Valid (17572 images)
    3. Test (33 images)
    4. 38 classes              
                """)
    
#prediction page

elif(app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    
    # Uploading the image
    test_image = st.file_uploader("Choose an image:")
    
    # Check if an image has been uploaded
    if test_image is not None:
        # Display the "Show Image" button
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        
        # Display the "Predict" button
        if st.button("Predict"):
            with st.spinner("Please Wait..."):
                time.sleep(3)
                st.write("### Our Prediction")
                result_index = model_prediction(test_image)
            
                #define class
                class_name= ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
                st.success("Model is predicting it is a {}".format(class_name[result_index]))
 
            