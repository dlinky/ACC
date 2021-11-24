# CBC with OpenCV

## 파일 구성

CBC.py : 메인 기능

get_roi.py : 스코프 검출&처리

manual_roi.py : 스코프 검출이 안되었을 경우 수동으로 검출

labelimg_xml.py : labelImg가 읽을 수 있도록 xml파일로 출력


## 필요 라이브러리 설치

cv2.connectedComponentsWithStatsWithAlgorithm() 사용하기 위해 opencv-python 버전 4.0 이상이어야 함

    conda install opencv-python

4.0버전에서는 cv2.CCL_SAUF, cv2.CCL_BBDT 두가지를 지원함. cv2.CCL_SPAGHETTI를 사용할 경우 python>3.9, opencv-python>4.5.3이어야 함

    conda install python=3.9
    pip install opencv-python

## 파일 경로
./original/ : 원본 이미지 파일

./result/ : 결과 xml 파일

./debug/ : 디버그 내용(프로세스, bounding box 처리 결과 이미지)

## 실행 방법
    CBC.py [-args]

'-d' : 디버그 모드(각 프로세스별 이미지 출력)

'-s' : 스코프 모드(스코프 제거)