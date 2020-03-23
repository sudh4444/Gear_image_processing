import cv2,numpy as np
from sklearn.linear_model import LinearRegression

def save_image(name,image):
    global count,local_index
    cv2.imwrite("analysis/"+str(count)+"_"+str(local_index)+"_"+str(name) +".jpg", image)
    local_index+=1

def linear_reg_sc(pos_y,pos_x):
	y = np.array(pos_y)
	x = np.array(pos_x).reshape(-1,1)
	model = LinearRegression().fit(x,y)
	print("model_intercept c:",model.intercept_)
	print("model_coef m:",model.coef_)
	y_new = model.predict(x)
	return y_new

def linear_reg(pos_y,pos_x):

	# pos_y, pos_x = np.where(block_img>=180)
	y_mean = 0
	x_mean = 0
	y_mean = np.mean(pos_y)
	x_mean = np.mean(pos_x)
	numerator,denominator,b1,b2 = 0,0,0,0

	for i in range(len(pos_y)):
		numerator+= (pos_x[i]-x_mean)*(pos_y[i]-y_mean)
		denominator+= (pos_x[i]-x_mean)**2

	b1 = numerator/denominator
	b2 = y_mean-(x_mean*b1)

	print("b1 = ",b1)
	print("b2 = ",b2)

	return b1,b2

if __name__ == "__main__":
	x = [1,2,3,4,5]
	y = [3,4,2,4,5]

	linear_reg(y,x)