import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 学習済みカスケードファイル一覧
cascade_alt_tree_path = 'haarcascade_frontalface_alt_tree.xml'
cascade_alt_path = 'haarcascade_frontalface_alt.xml'
cascade_alt2_path = 'haarcascade_frontalface_alt2.xml'
cascade_default_path = 'haarcascade_frontalface_default.xml'

# カスケード分類器をインスタンス化
cascade_alt_tree = cv2.CascadeClassifier(cascade_alt_path)
cascade_alt = cv2.CascadeClassifier(cascade_alt_path)
cascade_alt2 = cv2.CascadeClassifier(cascade_alt2_path)
cascade_default = cv2.CascadeClassifier(cascade_default_path)

# 辞書化
cascade_dict = {
    'alt_tree' : cascade_alt_tree,
    'alt' : cascade_alt,
    'alt2' : cascade_alt2,
    'default' : cascade_default
}

# タイトル
st.title('Open-CV-practice')

# サイドバー
selected_cascade = st.sidebar.selectbox('カスケード分類器を選択してください。',
    ['alt_tree', 'alt', 'alt2', 'default'])

# 識別に使うカスケード分類器
use_cascade = cascade_dict[selected_cascade]

selected_scale = st.sidebar.slider(label='scaleFactorの値を設定してください。',
                                    min_value=1.01,
                                    max_value= 2.0,
                                    value = 1.1)

selected_min_neighbors = st.sidebar.slider(label='minNeighborsの値を設定してください。',
                                min_value=1,
                                max_value=20,
                                value=2)

selected_min_size = st.sidebar.slider(label='minSizeの値を設定してください。',
                                        min_value=1,
                                        max_value=400,
                                        value = 50)

selected_angle = st.sidebar.slider(label='回転角の設定(左回転)',
                                    min_value= -180,
                                    max_value=180,
                                    value=0)

selected_color_name = st.sidebar.selectbox('描画する四角形の色を選択してください。',
                            ['赤', '白', '緑', '青', '黄色', '黒'])

color_dict = {
    '赤' : (255,0,0),
    '白' : (255,255,255),
    '緑' : (0,128,0),
    '青' : (0,0,255),
    '黒' : (0,0,0),
    '黄色' : (255,255,0)
}

selected_color = color_dict[selected_color_name]

# メインコンテンツ
col1, col2 = st.beta_columns(2)
with col1:
    st.write('以下の設定で顔認識を行います。')
    st.text(f'カスケード分類器: {selected_cascade}')
    st.text(f'scaleFactor: {selected_scale}')
    st.text(f'minNeighbors: {selected_min_neighbors}')
    st.text(f'minSize: ({selected_min_size}, {selected_min_size})')
    if selected_angle == 0:
        st.text('画像を回転させない')
    elif selected_angle > 0:
        st.text(f'画像を左に{selected_angle}° 回転' )
    elif selected_angle < 0:
        st.text(f'画像を右に{-selected_angle}° 回転' )

with col2:
    uploaded_file = st.file_uploader('画像をアップロードすると識別を開始します。', type=['jpg', 'jpeg', 'png'])

# 画像がアップロードされた時
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    np_img = np.array(img, dtype =np.uint8)
    img_gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    if selected_angle != 0:
        height, width = np_img.shape[:2]
        size = (height, width)
        angle_rad = selected_angle / 180.0*np.pi
        w_rot = int(np.round(height*np.absolute(np.sin(angle_rad))+width*np.absolute(np.cos(angle_rad))))
        h_rot = int(np.round(height*np.absolute(np.cos(angle_rad))+width*np.absolute(np.sin(angle_rad))))
        size_rot = (w_rot, h_rot)

        # 元画像の中心を軸にして画像を回転させる
        center = (width/2, height/2)
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, selected_angle, scale)

        # 平行移動を加える
        affine_matrix = rotation_matrix.copy()
        affine_matrix[0][2] = affine_matrix[0][2] -width/2 + w_rot/2
        affine_matrix[1][2] = affine_matrix[1][2] -height/2 + h_rot/2

        np_img = cv2.warpAffine(np_img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)
        img_gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

    face_list = use_cascade.detectMultiScale(img_gray, scaleFactor = selected_scale,
                                            minNeighbors = selected_min_neighbors,
                                            minSize = (selected_min_size, selected_min_size))
    face_number = len(face_list)
    if face_number == 0:
        st.error('顔が検出されませんでした。')
        st.image(np_img)
    if face_number > 0:
        for rect in face_list:
            cv2.rectangle(np_img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), selected_color, thickness = 4)
        st.image(np_img)