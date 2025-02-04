import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model  

def main():
    st.title('깨끗한 방인지 더러운 방인지!')
    st.info('방 사진을 업로드하면, 깨끗한 방인지 더러운 방인지 알려드립니다!')

    # 파일 업로더
    file = st.file_uploader('방 사진을 업로드하세요', type=['jpg', 'png', 'jpeg'])

    if file is not None:
        # 업로드된 이미지 표시 (use_column_width 대신 use_container_width 사용)
        st.image(file, caption="업로드한 방 사진", use_container_width=True)
        
        #  1. 모델 로드
        model = load_model('model/keras_model.h5', compile=False)
        
        #  2. 클래스 이름 로드 (📌 CP949 인코딩 적용)
        try:
            with open('model/labels.txt', 'r', encoding='cp949') as f:
                class_names = f.read().splitlines()
        except UnicodeDecodeError:
            # CP949로 안 될 경우 UTF-8로 다시 시도
            with open('model/labels.txt', 'r', encoding='utf-8') as f:
                class_names = f.read().splitlines()

        #  3. 이미지 전처리
        size = (224, 224)
        image = Image.open(file)  # 파일에서 이미지 로드
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)  # 크기 조정
        
        # 이미지 numpy 배열 변환 및 정규화
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        #  4. 예측 수행
        prediction = model.predict(data)
        predicted_index = np.argmax(prediction)  # 가장 높은 확률의 클래스 선택
        predicted_class = class_names[predicted_index]  # 해당 클래스명 가져오기
        confidence = prediction[0][predicted_index] * 100  # 확률 값 변환

        #  숫자 제거 후 클래스명 정리
        cleaned_class = predicted_class.split(" ", 1)[-1]  # "0 Clean" → "Clean", "1 Messy" → "Messy"

        #  5. 결과 출력
        st.success(f'예측 결과: 이 방은 [{cleaned_class}] 방입니다 ({confidence:.2f}%) 확률')

if __name__ == '__main__':
    main()
