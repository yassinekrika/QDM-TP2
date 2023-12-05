import tkinter
from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
import pandas as pd
import os
import pywt
import matplotlib.pyplot as plt 
from scipy.special import gamma
from scipy.stats import gennorm
from scipy.spatial.distance import cityblock


class App(tkinter.Tk):

    image1 = None
    image2 = None

    targer_folder = ""

    def __init__(self):
        super().__init__()
        self.title("TP1")
        self.geometry("500x500")
        self.maxsize(500, 500)
        self.minsize(500, 500)

        self.frame = Frame(self, bg='#41B77F')

        self.image1_label = Label(self, text="Image 1:")
        self.image1_label.grid(row=0, column=1, padx=10, pady=10)
        
        self.image1_button = Button(self, text="Upload Image 1", command=self.upload_image1)
        self.image1_button.grid(row=0, column=0, padx=10, pady=10)

        self.image2_label = Label(self, text="Image 2:")
        self.image2_label.grid(row=1, column=1, padx=10, pady=10)

        self.image2_button = Button(self, text="Upload Image 2", command=self.upload_image2)
        self.image2_button.grid(row=1, column=0, padx=10, pady=10)

        self.clear_button = Button(self, text="Clear Images", command=self.clear_images)
        self.clear_button.grid(row=2, column=0, pady=10)

        self.psnr_button = Button(self, text="Visual Quality of Image", command=self.calcualte_visual_quality)
        self.psnr_button.grid(row=4, column=0, pady=10)

        self.objective_result_button = Button(self, text="Objective Result", command=self.calculate_objective_result)
        self.objective_result_button.grid(row=7, column=0, pady=10)

        self.comparaison_button = Button(self, text="Make Comparaison", command=self.calculate_comparaison)
        self.comparaison_button.grid(row=8, column=0, pady=10)

        self.quit_button = Button(self, text="Quit", command=self.quit)
        self.quit_button.grid(row=9, column=0, pady=10)

    def upload_image1(self):
        self.image1 = filedialog.askopenfilename(initialdir = "/home/yassg4mer/Downloads/Py/",title = "Select file",filetypes = (("image files","*.bmp"),("all files","*.*")))
        self.image1_label = Label(self, text=self.image1)
        self.image1_label.grid(row=0, column=1, padx=10, pady=10)

        self.image1 = cv2.imread(self.image1, cv2.IMREAD_GRAYSCALE)
        self.image1 = self.image1.astype(np.float64)
        
    def upload_image2(self):
        self.image2 = filedialog.askopenfilename(initialdir = "/home/yassg4mer/Downloads/Py/",title = "Select file",filetypes = (("image files","*.bmp"),("all files","*.*")))
        self.image2_label = Label(self, text=self.image2)
        self.image2_label.grid(row=1, column=1, padx=10, pady=10)

        self.image2 = cv2.imread(self.image2, cv2.IMREAD_GRAYSCALE)
        self.image2 = self.image2.astype(np.float64)

    def clear_images(self):
        self.image1_label.destroy()
        self.image2_label.destroy()

    def calcualte_visual_quality(self):
        original_vector = self.process_original_image()
        destored_vector = self.process_destored_image()

        # Calculate city block distance between vectors
        distance = []
        for i in range(len(original_vector)):
            distance.append(self.calculate_city_block_distance(original_vector[i], destored_vector[i]))

        quality = self.quality(distance)
        self.quality_label = Label(self, text=str(quality))
        self.quality_label.grid(row=4, column=1, pady=10)

        print('quality', quality)
        
    def estimate_GGD_parameters(self, vec):
        gam =np.arange(0.2, 10.0, 0.001)#Generate candidate γ
        r_gam = (gamma(1/gam)*gamma(3/gam))/((gamma(2/gam))**2)# Calculate r(γ) from candidate γ
        sigma_sq=np.mean((vec)**2)#σ^2 zero mean estimation,non-zero mean needs to calculate the mean and then according to(3)
        sigma=np.sqrt(sigma_sq)
        E=np.mean(np.abs(vec))
        r=sigma_sq/(E**2)#calculate r(γ) from sigma and E
        diff=np.abs(r-r_gam)
        alpha=gam[np.argmin(diff, axis=0)]
        beta = sigma * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))

        return alpha, beta

    def GGD(self, x, alpha, beta):

        coefficien=alpha/(2*beta*gamma(1/alpha))

        return coefficien*np.exp((-(np.abs(x)/beta)**alpha))

    def quality(self, subands):
        somme=0
        for dcity in subands:
            somme += np.log10(1 + dcity)
        return (1/len(subands)) * somme

    def process_original_image(self):
        level = 3
        coeffs_original_image = pywt.wavedec2(self.image1, 'db1', level=level)
        vector_shape = []


        # Print each subband and estimate GGD parameters
        for i in range(1, level + 1):
            for j in range(3):
                alpha, beta = self.estimate_GGD_parameters(coeffs_original_image[i][j])

                x = np.linspace(min(coeffs_original_image[i][j].flatten()), max(coeffs_original_image[i][j].flatten()), 100)
                ggd_curve = gennorm.pdf(x, alpha, 0, beta)
                vector_shape.append(ggd_curve)

        return vector_shape

    def process_destored_image(self):
        level = 3
        coeffs_destored_image = pywt.wavedec2(self.image2, 'db1', level=level)
        vector_shape = []

        for i in range(1, level + 1):
            for j in range(3):
                hist, bins = np.histogram(coeffs_destored_image[i][j], bins='auto', density=True)

                shape, loc, scale = gennorm.fit(coeffs_destored_image[i][j])

                ggd_pdf = gennorm.pdf(bins, shape, loc, scale)
                vector_shape.append(ggd_pdf)

        return vector_shape

    def calculate_city_block_distance(self, vector1, vector2):

        min_len = min(len(vector1), len(vector2))
        vector1_resampled = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(vector1)), vector1)
        vector2_resampled = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(vector2)), vector2)


        distance = cityblock(vector1_resampled, vector2_resampled)
        distance2 = np.linalg.norm(vector1_resampled - vector2_resampled, ord=1)

        return distance

    def calculate_objective_result(self):
        self.targer_folder = filedialog.askdirectory(initialdir = "/home/yassg4mer/Downloads/Py/",title = "Select folder")
        self.open_folder()

    def open_folder(self):

        info_file = self.targer_folder + '/info.txt'
        with open(info_file) as f:
            lines = f.readlines()

            image_origin_array = []
            image_degraded_array = []

            ssim_result_array = []
            ms_ssim_result_array = []
            psnr_result_array = []

            for line in lines:
                image_origin = line.split(' ')[0]
                image_degraded = line.split(' ')[1]

                image_origin_array.append(image_origin)
                image_degraded_array.append(image_degraded)

                image_origin_path = '/home/yassg4mer/Downloads/Py/refimgs/' + image_origin
                image_degraded_path = self.targer_folder + '/' + image_degraded
                print(image_degraded_path)

                x = cv2.imread(image_origin_path, cv2.IMREAD_GRAYSCALE)
                y = cv2.imread(image_degraded_path, cv2.IMREAD_GRAYSCALE)

                x = x.astype(np.float64)
                y = y.astype(np.float64)

                ssim_result = self.ssim(x, y)
                ms_ssim_result = self.ms_ssim(x, y)
                psnr_result = self.psnr(x, y)

                ms_ssim_result_array.append(ms_ssim_result)
                ssim_result_array.append(ssim_result)
                psnr_result_array.append(psnr_result)

                print(image_origin, image_degraded, ssim_result, ms_ssim_result, psnr_result)
                
            df = pd.DataFrame({'image_origin': image_origin_array, 
                                'image_degraded': image_degraded_array, 
                                'ssim_result': ssim_result_array, 
                                'ms_ssim_result': ms_ssim_result_array, 
                                'psnr_result': psnr_result_array
                                })
            df.to_excel('objective.xlsx', index=False)

            os.system('libreoffice --calc objective.xlsx')

    def pcc(self, img1, img2):

        mean_X = np.mean(img1)
        mean_Y = np.mean(img2)

        covariance = np.sum((img1 - mean_X) * (img2 - mean_Y))

        s_X = np.sqrt(np.sum((img1 - mean_X)**2))
        s_Y = np.sqrt(np.sum((img2 - mean_Y)**2))

        pcc = covariance / (s_X * s_Y)

        return pcc

    def rho(self, img1, img2):
        n = len(img1)
        
        ranked_data1 = sorted(range(n), key=lambda i: img1[i])
        ranked_data2 = sorted(range(n), key=lambda i: img2[i])

        d = [rank1 - rank2 for rank1, rank2 in zip(ranked_data1, ranked_data2)]

        rho = 1 - (6 * sum(x**2 for x in d)) / (n * (n**2 - 1))

        return rho

    def calculate_comparaison(self):
        self.open_comparaison_folder()
    
    def open_comparaison_folder(self):
        df = pd.read_excel('/home/yassg4mer/Downloads/Py/objective.xlsx', usecols=['ssim_result', 'ms_ssim_result', 'psnr_result'] )
        dff = pd.read_excel('/home/yassg4mer/Downloads/Py/subjective.xlsx', usecols=['subjective_result'] )


        rmse_psnr_result_array = []
        pcc_psnr_result_array = []
        rho_psnr_result_array = []

        rmse_ssim_result_array = []
        pcc_ssim_result_array = []
        rho_ssim_result_array = []

        rmse_ms_ssim_result_array = []
        pcc_ms_ssim_result_array = []
        rho_ms_ssim_result_array = []


        # psnr result
        rmse_psnr_result = self.rmse(df['psnr_result'], dff['subjective_result'])
        pcc_psnr_result = self.pcc(df['psnr_result'], dff['subjective_result'])
        rho_psnr_result = self.rho(df['psnr_result'], dff['subjective_result'])

        # ssim result
        rmse_ssim_result = self.rmse(df['ssim_result'], dff['subjective_result'])
        pcc_ssim_result = self.pcc(df['ssim_result'], dff['subjective_result'])
        rho_ssim_result = self.rho(df['ssim_result'], dff['subjective_result'])

        # ms_ssim result
        rmse_ms_ssim_result = self.rmse(df['ms_ssim_result'], dff['subjective_result'])
        pcc_ms_ssim_result = self.pcc(df['ms_ssim_result'], dff['subjective_result'])
        rho_ms_ssim_result = self.rho(df['ms_ssim_result'], dff['subjective_result'])

        rmse_psnr_result_array.append(rmse_psnr_result)
        pcc_psnr_result_array.append(pcc_psnr_result)
        rho_psnr_result_array.append(rho_psnr_result)

        rmse_ssim_result_array.append(rmse_ssim_result)
        pcc_ssim_result_array.append(pcc_ssim_result)
        rho_ssim_result_array.append(rho_ssim_result)

        rmse_ms_ssim_result_array.append(rmse_ms_ssim_result)
        pcc_ms_ssim_result_array.append(pcc_ms_ssim_result)
        rho_ms_ssim_result_array.append(rho_ms_ssim_result)

        dfff = pd.DataFrame({ # psnr
                    'rmse_psnr_result': rmse_psnr_result_array, 
                    'pcc_psnr_result': pcc_psnr_result_array,
                    'rho_psnr_result': rho_psnr_result_array, 
                    # ssim
                    'rmse_ssim_result': rmse_ssim_result_array,
                    'pcc_ssim_result': pcc_ssim_result_array,
                    'rho_ssim_result': rho_ssim_result_array,
                    # ms_ssim
                    'rmse_ms_ssim_result': rmse_ms_ssim_result_array,
                    'pcc_ms_ssim_result': pcc_ms_ssim_result_array,
                    'rho_ms_ssim_result': rho_ms_ssim_result_array,})

        dfff.to_excel('comparaison.xlsx', index=False)


        os.system('libreoffice --calc comparaison.xlsx')

    def quit(self):
        self.destroy()

if __name__=="__main__":
    app = App()
    app.mainloop()