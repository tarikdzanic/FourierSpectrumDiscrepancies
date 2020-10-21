Supplementary codes for the paper:
	Fourier Spectrum Discrepancies in Deep Network Generated Images
	Tarik Dzanic, Karan Shah, and Freddie D. Witherden
	In NeurIPS, 2020


Two MATLAB codes are attached in this supplementary material along with a portion of the dataset for reproducing similar results. 
- generatedata.m
- classify.m

Dependencies: 
- MATLAB

Instructions:
- Run generatedata.m using MATLAB. This code calculates the reduced spectra of the included example images (5 per model, Q100, 256/768/1024 resolution) and 
	fits the decay parameters to the spectra.

- Run classify.m using MATLAB. The fit coefficients for a portion of the entire dataset are included in the folder Fits. The classifier outputs the overall classification
	accuracy as well as the accuarcy for classifying one model vs. real images at a time. Slight discrepencies in the classification accuracies are seen since the 
	dataset is reduced. 


