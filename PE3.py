import numpy as np
import cv2 as cv

def main():
	img_in = cv.imread('Screenshot 2023-04-05 054153.png', cv.IMREAD_GRAYSCALE)
	roi = img_in[0:(img_in.shape[0] & -2), 0:(img_in.shape[1] & -2)]
	h = calcPSF(roi.shape, 53)
	Hw = calcWnrFilter(h, 1.0 / float(5200))
	imgOut = filter2DFreq(roi, Hw)
	imgOut = imgOut.astype(np.float32)
	imgOut = cv.normalize(imgOut, imgOut, alpha=0, beta=255,
		norm_type=cv.NORM_MINMAX)
	cv.imwrite("Deblur.jpg", imgOut)

def calcPSF(filterSize, R):
	h = np.zeros(filterSize, dtype=np.float32)
	point = (filterSize[1] // 2, filterSize[0] // 2)
	h = cv.circle(h, point, R, 255, -1, 8)
	summa = np.sum(h)
	return (h / summa)

def filter2DFreq(inputImg, H):
	planes = [inputImg.copy().astype(np.float32),
		np.zeros(inputImg.shape, dtype=np.float32)]
	complexI = cv.merge(planes)
	complexI = cv.dft(complexI, flags=cv.DFT_SCALE)
	planesH = [H.copy().astype(np.float32),
		np.zeros(H.shape, dtype=np.float32)]
	complexH = cv.merge(planesH)
	complexIH = cv.mulSpectrums(complexI, complexH, 0)
	complexIH = cv.idft(complexIH)
	planes = cv.split(complexIH)
	return planes[0]

def calcWnrFilter(input_h_PSF, nsr):
	h_PSF_shifted = np.fft.fftshift(input_h_PSF)
	planes = [h_PSF_shifted.copy().astype(np.float32),
		np.zeros(h_PSF_shifted.shape, dtype=np.float32)]
	complexI = cv.merge(planes)
	complexI = cv.dft(complexI)
	planes = cv.split(complexI)
	denom = np.power(np.abs(planes[0]), 2)
	denom += nsr
	return cv.divide(planes[0], denom)

if __name__ == "__main__":
	main()
	cv.destroyAllWindows()
	img = cv.imread('Deblur.jpg')
	img = cv.resize(img,(600,400))
	cv.imshow('',img)
	cv.waitKey(0)