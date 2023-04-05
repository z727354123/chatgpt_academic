import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator, image_utils
from PIL import Image

img_to_array = image_utils.img_to_array
load_img = image_utils.load_img

if __name__ == '__main__':

	# 定义模型
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(640, 640, 3)))
	model.add(MaxPooling2D((2, 2), padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2), padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2), padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2), padding='same'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

	# 编译模型
	model.compile(optimizer='adam', loss='mse')

	# 加载图像
	img = load_img('/Users/judy/Downloads/123_png.png')
	x = img_to_array(img)
	# 创建形状为 (636,636,3) 的随机数组
	# 创建形状为 (640,630,3) 的零数组
	new_a = np.zeros((640, 640, 3))

	# 将原始数组值复制到新数组中
	new_a[:636, :636, :] = x
	x = new_a
	x = x.reshape((1,) + x.shape)
	x = x / 255.0

	# 数据增强
	datagen = ImageDataGenerator(
		rotation_range=45,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest'
	)

	# 训练模型
	for i in range(10):
		print("Training epoch", i + 1)
		datagen.fit(x)
		model.fit_generator(datagen.flow(x, x, batch_size=1), steps_per_epoch=len(x), epochs=1)

	# 修复图像
	y = model.predict(x)
	y = y * 255.0
	y = y.astype('uint8')
	result = Image.fromarray(y[0])

	# 显示修复后的图像
	result.show()