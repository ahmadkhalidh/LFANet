import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import tempfile
import os

# ---------- GEMINI ----------
import google.generativeai as genai
genai.configure(api_key= "")

st.title("Liver Fibrosis Classification with Grad-CAM")

CLASS_NAMES = ["F0", "F1", "F2", "F3", "F4"]
LAST_CONV_LAYER = "conv2d_10"   

# -----------------------------------------------------------
# SELF-ATTENTION LAYER
# -----------------------------------------------------------

def self_attention(input_feature_map):
    shape = K.int_shape(input_feature_map)
    _, height, width, channels = shape

    flat = Reshape((height * width, channels))(input_feature_map)

    q1 = Dense(channels // 4, activation='relu')(flat)
    k1 = Dense(channels // 4, activation='relu')(flat)
    v1 = Dense(channels // 2, activation='relu')(flat)

    q = Dense(channels // 8)(q1)
    k = Dense(channels // 8)(k1)
    v = Dense(channels)(v1)

    kt = Permute((2, 1))(k)

    attention_scores = Lambda(lambda x: K.batch_dot(x[0], x[1]))([q, kt])
    attention_scores = Softmax()(attention_scores)

    attention_output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attention_scores, v])
    attention_output = Reshape((height, width, channels))(attention_output)

    return Add()([input_feature_map, attention_output])


# -----------------------------------------------------------
# REBUILD TRAINING ARCHITECTURE
# -----------------------------------------------------------

def build_model():
    input_layer = tf.keras.Input(shape=(128, 128, 3))

    x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D(3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x1 = Conv2D(128, 3, dilation_rate=1, padding='same', activation='relu')(x)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(128, 3, dilation_rate=3, padding='same', activation='relu')(x)
    x2 = BatchNormalization()(x2)
    x3 = Conv2D(128, 3, dilation_rate=5, padding='same', activation='relu')(x)
    x3 = BatchNormalization()(x3)

    A1 = Add()([x1, x2, x3])

    x4 = Conv2D(128, 3, padding='same', activation='relu')(A1)
    x4 = BatchNormalization()(x4)

    x = self_attention(x4)

    C1 = Concatenate()([x, x2, x3])
    x5 = Conv2D(128, 3, padding='same', activation='relu')(C1)

    C2 = Concatenate()([x5, x])
    x = Conv2D(256, 3, padding='same', activation='relu')(C2)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D(3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, 1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    residual = Conv2D(256, 1, padding='same', activation='relu')(C2)
    x = Add()([x, residual])

    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(5, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=output)


# -----------------------------------------------------------
# LOAD WEIGHTS
# -----------------------------------------------------------

@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("livermodel.h5")
    return model

model = load_model()

# -----------------------------------------------------------
# GRAD-CAM
# -----------------------------------------------------------
def make_gradcam(img):
    grad_model = Model(model.inputs,[model.get_layer(LAST_CONV_LAYER).output,model.output])
    with tf.GradientTape() as tape:
        conv, pred = grad_model(img)
        idx = tf.argmax(pred[0])
        loss = pred[:,idx]
    grads = tape.gradient(loss, conv)[0]
    pooled = tf.reduce_mean(grads,axis=(0,1,2))
    conv = conv[0]
    heat = tf.reduce_sum(conv * pooled,axis=-1).numpy()
    heat = np.maximum(heat,0) / np.max(heat)
    return heat, int(idx)

def gradcam_stats(h):
    return {
        "mean_activation": float(np.mean(h)),
        "max_activation": float(np.max(h)),
        "high_attention_ratio": float(np.mean(h > 0.6))
    }

# -----------------------------------------------------------
# GEMINI EXPLANATION
# -----------------------------------------------------------
def explain_with_gemini(label, confidence, stats):
    prompt = f"""
You are explaining the output of an AI system to a general audience with no medical background.

A computer system analyzed an ultrasound image of the liver and produced these results:

- Estimated fibrosis level: {label}
- Confidence in this estimate: {confidence:.3f}

The system also highlighted areas of the image that influenced its decision.

Please explain the results in simple, reassuring language:

1. **What this fibrosis level means**  
Explain what this level generally indicates about how liver tissue may appear on ultrasound images, using plain language and avoiding medical jargon.

2. **What the highlighted image areas mean**  
Describe, in simple terms, why certain parts of the image were highlighted and what that tells us about how the system looked at the image.

3. **How doctors usually view these image patterns**  
Explain how medical professionals commonly use ultrasound images to observe liver structure, without giving advice or conclusions.

Important constraints:
- Do not provide a diagnosis
- Do not recommend treatment
- Do not give medical advice
- Make it clear that this explanation is generated by a computer system and is not a medical opinion
- Avoid technical terms unless they are clearly explained

Write in a calm, supportive tone.
Use short sentences and short paragraphs.
Target length: about 200 words.
"""
    response = gemini.generate_content(prompt)
    return response.text.strip()

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
uploaded = st.file_uploader("Upload liver ultrasound image")

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    arr = cv2.resize(np.array(img),(128,128)).astype("float32")/255.0
    arr = np.expand_dims(arr,0)

    preds = model.predict(arr)[0]
    cls_id = np.argmax(preds)
    conf = preds[cls_id]

    st.subheader("Prediction")
    st.write(CLASS_NAMES[cls_id], f"(confidence {conf:.3f})")

    heat, _ = make_gradcam(arr)
    stats = gradcam_stats(heat)

    orig = cv2.resize(np.array(img),(128,128))
    heat_r = cv2.resize(heat,(128,128))
    overlay = cv2.addWeighted(
        cv2.applyColorMap(np.uint8(255*heat_r),cv2.COLORMAP_JET),
        0.4, orig, 0.6, 0
    )

    col1,col2,col3 = st.columns(3)
    col1.image(orig,width=300,caption="Original")
    col2.image(heat_r,width=300,caption="Heatmap")
    col3.image(overlay,width=300,caption="Overlay")

    if st.button("Generate Interpretation"):
        with st.spinner("Generating explanation..."):
            explanation = explain_with_gemini(CLASS_NAMES[cls_id],conf,stats)
        st.write(explanation)
