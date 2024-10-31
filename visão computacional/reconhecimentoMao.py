from cvzone.HandTrackingModule import HandDetector
import cv2

# Inicializa a webcam para capturar vídeo
# O '2' indica a terceira câmera conectada ao seu computador; '0' normalmente se refere à câmera embutida
cap = cv2.VideoCapture(0)

# Inicializa a classe HandDetector com os parâmetros fornecidos
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# Continuamente captura frames da webcam
while True:
    # Captura cada frame da webcam
    # 'success' será True se o frame for capturado com sucesso, 'img' conterá o frame
    success, img = cap.read()

    # Encontra as mãos no frame atual
    # O parâmetro 'draw' desenha os pontos de referência e contornos da mão na imagem se configurado como True
    # O parâmetro 'flipType' inverte a imagem, facilitando algumas detecções
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Verifica se alguma mão foi detectada
    if hands:
        # Informações da primeira mão detectada
        hand1 = hands[0]  # Obtém a primeira mão detectada
        lmList1 = hand1["lmList"]  # Lista de 21 pontos de referência para a primeira mão
        bbox1 = hand1["bbox"]  # Caixa delimitadora ao redor da primeira mão (coordenadas x, y, w, h)
        center1 = hand1['center']  # Coordenadas do centro da primeira mão
        handType1 = hand1["type"]  # Tipo da primeira mão ("Esquerda" ou "Direita")

        # Conta o número de dedos levantados na primeira mão
        fingers1 = detector.fingersUp(hand1)
        print(f'H1 = {fingers1.count(1)}', end=" ")  # Imprime a contagem de dedos levantados

        # Calcula a distância entre pontos de referência específicos na primeira mão e desenha na imagem
        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),
                                                  scale=10)

        # Verifica se uma segunda mão foi detectada
        if len(hands) == 2:
            # Informações da segunda mão
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            center2 = hand2['center']
            handType2 = hand2["type"]

            # Conta o número de dedos levantados na segunda mão
            fingers2 = detector.fingersUp(hand2)
            print(f'H2 = {fingers2.count(1)}', end=" ")

            # Calcula a distância entre os dedos indicadores de ambas as mãos e desenha na imagem
            length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
                                                      scale=10)

        print(" ")  # Nova linha para melhor legibilidade da saída impressa

    # Exibe a imagem em uma janela
    cv2.imshow("Imagem", img)

    # Mantém a janela aberta e a atualiza para cada frame; espera 1 milissegundo entre os frames
    cv2.waitKey(1)
