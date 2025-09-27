import cv2 as cv  
import numpy as np  
import os  # OS para manipulação de caminhos
import glob


# Caminhos para os dados
img_folder = 'cat_dog_savi'  # Pasta onde estão as imagens e o ficheiro de labels
labels_path = os.path.join(img_folder, 'labels.txt')  # Caminho para o ficheiro de labels
img_filenames = glob.glob(os.path.join(img_folder, '*.jpeg'))  # Lista todos os ficheiros .jpeg na pasta

# Leitura dos labels (um por linha, ignorando linhas vazias)
with open(labels_path, 'r') as f:
	labels = [line.strip() for line in f if line.strip()]

print (str(labels))

# Lista dos nomes dos ficheiros de imagem (assumindo nomes 1.jpeg, 2.jpeg, ...)
img_files = [f"{i+1}.jpeg" for i in range(len(labels))]

# Tamanho para o qual todas as imagens serão redimensionadas (largura, altura)
target_size = (256, 256)

# Leitura e redimensionamento das imagens
imgs = []
for fname in img_files:
	img = cv.imread(os.path.join(img_folder, fname))  # Lê a imagem
	if img is not None:
		img = cv.resize(img, target_size)  # Redimensiona
	imgs.append(img)

# Mostra todas as imagens com o respetivo label
for idx, (img, label) in enumerate(zip(imgs, labels)):
	if img is not None:
		# Mostra a imagem numa janela com o nome e label
		cv.imshow(f"Image {idx+1} - {label}", img)
		print(f"Image {idx+1}: {label}")
	else:
		print(f"Image {idx+1}: Não foi possível carregar a imagem.")

# Mensagem para o utilizador
print("\nAs imagens estão abertas. Feche todas as janelas para continuar...")
# Espera até o utilizador pressionar uma tecla em qualquer janela
cv.waitKey(0)
# Fecha todas as janelas abertas do OpenCV
cv.destroyAllWindows()



# Função de classificação baseada em características visuais
def classify_image(img):
	# Converter a imagem para o espaço de cor HSV (útil para detetar cores) - Matriz (cor pura),
	# Saturação (intensidade da cor, quanto menor, mais cinza), Valor (claridade)
	hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	#cv.imshow("HSV", hsv)
	#cv.waitKey(0)

	# Definir a faixa de valores HSV que corresponde ao verde típico da relva
	lower_green = np.array([35, 40, 40])  # Limite inferior (H, S, V)
	upper_green = np.array([85, 255, 255])  # Limite superior (H, S, V)

	# Criar uma máscara binária onde os pixels dentro da faixa de verde são 1 (255 - branco)
	mask_green = cv.inRange(hsv, lower_green, upper_green)

	# Calcular a razão de pixeis verdes na imagem
	green_ratio = np.sum(mask_green > 0) / (img.shape[0] * img.shape[1])
	#print(f"Green ratio: {green_ratio:.2f}")

	# Converter para escala de cinzentos para analisar claridade e uniformidade
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	desvio_gray = np.std(gray)  # Desvio padrão: mede variação de intensidade
	mean_gray = np.mean(gray)  # Média: mede claridade geral
	#cv.imshow("Gray", gray)
	#cv.waitKey(0)
	print(f"Mean gray: {mean_gray:.2f}, Desvio gray: {desvio_gray:.2f}")

	# --- Classificação ---
	# - Se a imagem tem muito verde, é um gato
	# - Se o fundo é claro e uniforme, é um cão
	if green_ratio > 0.10:
		return 'cat'  # Muita relva
	elif mean_gray > 160 and desvio_gray < 35:
		return 'dog'  # Fundo claro e uniforme


# Apresenta os resultados da classificação para cada imagem
print("\n--- Resultados da Classificação ---")
for idx, img in enumerate(imgs):
	if img is not None:
		pred = classify_image(img)
		print(f"Image {idx+1}: Previsão: {pred} | Verdadeiro: {labels[idx]}")
	else:
		print(f"Image {idx+1}: Não foi possível carregar a imagem.")

# Garante que todas as janelas OpenCV são fechadas
cv.destroyAllWindows()