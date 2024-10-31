from cvzone.FaceMeshModule import FaceMeshDetector
import cv2

# Inicializa a webcam
# '2' indica a terceira câmera conectada ao computador, '0' normalmente se refere à webcam embutida
cap = cv2.VideoCapture(0)

# Inicializa o objeto FaceMeshDetector
# staticMode: Se True, a detecção ocorre apenas uma vez, caso contrário, em todos os frames
# maxFaces: Número máximo de rostos a serem detectados
# minDetectionCon: Limite mínimo de confiança para a detecção
# minTrackCon: Limite mínimo de confiança para o rastreamento
detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

# Inicia o loop para capturar continuamente frames da webcam
while True:
    # Lê o frame atual da webcam
    # success: Booleano, indica se o frame foi capturado com sucesso
    # img: O frame atual
    success, img = cap.read()

    # Encontra a malha facial na imagem
    # img: Imagem atualizada com a malha facial se draw=True
    # faces: Informações dos rostos detectados
    img, faces = detector.findFaceMesh(img, draw=True)

    # Verifica se algum rosto foi detectado
    if faces:
        # Loop através de cada rosto detectado
        for face in faces:
            # Obtém pontos específicos do olho
            # leftEyeUpPoint: Ponto acima do olho esquerdo
            # leftEyeDownPoint: Ponto abaixo do olho esquerdo
            leftEyeUpPoint = face[159]
            leftEyeDownPoint = face[23]

            # Calcula a distância vertical entre os pontos do olho
            # leftEyeVerticalDistance: Distância entre os pontos acima e abaixo do olho esquerdo
            # info: Informações adicionais (como coordenadas)
            leftEyeVerticalDistance, info = detector.findDistance(leftEyeUpPoint, leftEyeDownPoint)

            # Imprime a distância vertical para depuração ou informação
            print(leftEyeVerticalDistance)

    # Exibe a imagem em uma janela chamada 'Imagem'
    cv2.imshow("Imagem", img)

    # Espera 1 milissegundo para verificar se há alguma entrada do usuário, mantendo a janela aberta
    cv2.waitKey(1)


    # Detecta rostos na imagem
    # img: Imagem atualizada
    # bboxs: Lista de caixas delimitadoras ao redor dos rostos detectados
    img, bboxs = detector.findFaceMesh(img, draw=False)

    # Verifica se algum rosto foi detectado
    if bboxs:
        # Loop através de cada caixa delimitadora
        for bbox in bboxs:
            # bbox contém 'id', 'bbox', 'score', 'center'

            # ---- Obter Dados  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

            # ---- Desenhar Dados  ---- #
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, f'{score}%', (x, y - 10))
            cvzone.cornerRect(img, (x, y, w, h))

    # Exibe a imagem em uma janela chamada 'Imagem'
    cv2.imshow("Imagem", img)
    # Espera 1 milissegundo e mantém a janela aberta
    cv2.waitKey(1)
