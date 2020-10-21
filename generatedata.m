close all
clear all

methods  = ["ALAE", "PGGAN", "Real", "StyleGAN", "StyleGAN2", "VQVAE2"];
nexamples = 5;      % Number of example images 
rthresh = 0.50;      % Threshold above to azimuthally average Fourier coefficients
rthreshfit = 0.75;   % Threshold above to fit coefficients
nbins = 200;         % Number of bins for binning Fourier coefficients   
nsmooth = 5;         % Smoothing window for binned coefficient
pnorm = 2;           % Norm for decay fitting

%% Calculate FFT spectrum and decay fits for example images
for i=1:length(methods)
    method = methods(i)
    for j=0:nexamples-1
        for res=[256, 768, 1024]
            % Read image, compute radial Fourier coefficients, and bin
            % radial spectrum
            img = imread(['.\Datasets\Q100\', char(method), '\', num2str(res) ,'\', num2str(j), '.png']); 
            [r, m, dc] = processImage(rgb2gray(img), rthresh);
            [xh,yh] = binSpectrum(r,m,nbins,rthresh); 
            
            % Fit power law to binned spectrum
            c = getFitCoeffs(xh, yh, dc, nsmooth, 0.85, pnorm);
            save(['.\ExampleFits\Q100\', char(method), '\', num2str(res) ,'\', num2str(j), '.mat'], 'c');
        end
    end
end

  
% Compute radial Fourier spectrum of image above some radial threshold
function [r, m, dc] = processImage(img, rthresh)
    F = fftshift(fft2(img));    
    nx = int32(size(F,1)/2);
    ny = int32(size(F,2)/2);
    r = sqrt(double(nx)^2 + double(ny)^2);

    rlist = [];
    maglist = [];

    for i = 1:size(F,1)
        for j = 1:size(F,2)
            r_ij = sqrt(double(i-nx)^2 + double(j-ny)^2);
            if (r_ij > rthresh*r)
                rlist(length(rlist)+1) = r_ij;
                if (~isnan(abs(F(i,j))))
                    maglist(length(maglist)+1) = abs(F(i,j));
                else
                    maglist(length(maglist)+1) = 0;
                end
            end 
            if (r_ij == 0)
                dc = abs(F(i,j)); % Compute DC gain at r = 0
            end
        end
    end
    r = rlist;
    m = maglist;
end

function [xh,yh] = binSpectrum(x, y, nbins, rthresh)
    dx = max(x) - min(x);
    xmin = min(x); 
    xmax = max(x);
    ny = length(y);
    
    % Binned x,y
    xh = linspace(xmin, xmax, nbins + 1);
    yh = zeros(nbins,1);    nh = zeros(nbins,1);
    
    % Compute bins by rolling average
    for i=1:ny
        xcurr = x(i);
        for j=1:nbins
            if (xcurr >= xh(j) && xcurr <= xh(j+1))
                yh(j) = yh(j) + y(i);
                nh(j) = nh(j) + 1;
                break
            end
        end
    end
    
    % Set binned radius (newxh) to average of adjacent points
    yh = yh./nh;
    newxh = [];
    for i=1:nbins
        newxh(i) = (xh(i) + xh(i+1))/2.0;
    end
    
    % If binned coeff == 0, set to average of adjacent points since
    % plotting on logy scale
    for i=1:nbins
        if yh(i) == 0
            yh(i) = 0.5*(yh(i-1) + yh(i+1));
        end
    end
    xh = newxh;
end

% Fit power law decay to binned spectrum
function c = getFitCoeffs(x, y, dc, nsmooth, rthresh, pnorm)   
    % Offset from threshold = 0.5 to fitting threshold (using 200 bins)
    nstart = int32(200*(rthresh - 0.5)/0.5) + 1;
    ystart = y(1);
    y = smooth(y,nsmooth);
    x = x(nstart:end)/max(x);
    y = y(nstart:end);

    yi = y(1);
    yf = y(end);

    g = fitPower(x,y, yi, pnorm); 
    c = [g(1), yi, yf];   
end


function C = fitPower(x,y,yi, pnorm)
    f = @(c,x) yi*((x/x(1)).^c(1));   % Objective Function
    options = optimset( 'MaxFunEvals', 99000000);
    C = fminsearch(@(c) norm(y - f(c,x), pnorm), [-2],options);
end