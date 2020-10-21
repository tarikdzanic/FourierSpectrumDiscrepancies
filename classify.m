clear all
close all

methods  = ["Real", "ALAE", "PGGAN", "StyleGAN", "StyleGAN2", "VQVAE2"];
qpath = 'Q100';   % Quality level [Q100, Q95, Q85]
res = 1024;       % Image resolution [1024, 768, 256]
k = 5;            % K-nearest neighbors parameter



realidx = 1; % Index of real dataset
% Set classification label to 0 for fake and 1 for real
classrf = zeros(length(methods),1); 
classrf(realidx) = 1;
% Tag image types
classtyp = [1, 2, 3, 4, 5, 6];

% Set number of training and total examples based on resolution
if res == 256
    nexamples = [1000, 1000, 1000, 999, 1000, 364];   
    ntrain =    [ 100,  100,  100, 100,  100, 100]; 
else
    nexamples = [1000, 1000, 1000, 1000, 1000, 12];  
    ntrain =    [ 100,  100,  100,  100,  100,  8];  
end


indivacc = zeros(length(ntrain), 1);
for i=1:length(ntrain)
    % If i = realidx, run classification for all methods at once, else run
    % one method vs. real at a time    
    if i ~= realidx
        name = [char(methods(i)), ' vs. Real Acc. = '];
    else    
        name = ['Overall Acc. = '];
    end
    % Run classifier
    [acc, typacc] = runKNN(k, res, methods, classrf, classtyp, nexamples, ntrain, qpath, i, realidx);
    indivacc(i) = acc*100;

    disp([name, num2str(indivacc(i)), '%'])
end


% Run KNN classifier, return overall accuracy and individual accuracy
function [acc, typacc] = runKNN(k, res, methods, classrf, classtyp, nexamples, ntrain, qpath, typidx, realidx)
    nprojtrain = zeros(length(ntrain), 1);
    nprojtest  = zeros(length(ntrain), 1);
    
    if typidx ~= realidx % If not running classification for all methods
        nprojtrain(realidx) = max(ntrain(realidx), ntrain(typidx)); % Copy training examples if less than highest number of examples
        nprojtrain(typidx) = max(ntrain(realidx), ntrain(typidx));

        nprojtest(realidx) = max(nexamples(realidx), nexamples(typidx)); % Copy testing examples if less than highest number of examples
        nprojtest(typidx) = max(nexamples(realidx), nexamples(typidx));
    else
        nprojtrain(:) = max(ntrain);
        nprojtest(:) = max(nexamples);
    end

    
    Xtrain = []; Ytrain = []; Ctrain = [];
    Xtest = []; Ytest = []; Ctest = [];
    for i=1:length(methods)
        method = methods(i);
        
        % Repeat until same number of images for each dataset
        n = 0;
        while n < nprojtrain(i)
            for j=0:ntrain(i)-1
                load(['.\Fits\', qpath,'\', char(method),  '\', num2str(res), '\', num2str(j), '.mat'], 'c');
                % X,Y,C = fit coeffs, real/fake, model type
                Xtrain = [Xtrain; c];
                Ytrain = [Ytrain; classrf(i)];
                Ctrain = [Ctrain; classtyp(i)];
                
                n = n + 1;
                if n == nprojtrain(i)
                    break
                end
                
            end
        end
        
        n = 0;
        while n < nprojtest(i)
            for j=ntrain(i)-1:nexamples(i)-1       
                load(['.\Fits\', qpath,'\', char(method),  '\', num2str(res), '\', num2str(j), '.mat'], 'c');
                Xtest = [Xtest; c];
                Ytest = [Ytest; classrf(i)];
                Ctest = [Ctest; classtyp(i)];
                n = n + 1;
                if n == nprojtest(i)
                    break
                end
            end
        end

    end

    classifier = fitcknn(Xtrain,Ytrain,'NumNeighbors',k,'Standardize',1);
    [acc, typacc] = computeAccuracy(classifier, Xtest, Ytest, Ctest);
end


% Compute accuracy based on hits and misses
function [acc, typacc] = computeAccuracy(classifier, Xtest, Ytest, Ctest)
    % For all methods
    hit = 0;    miss = 0;
    % For individual methods
    hittyp = zeros(max(Ctest),1);    misstyp = zeros(max(Ctest),1);
    
    for i=1:length(Ytest)
        x = Xtest(i,:);
        yhat = predict(classifier, x);
        if yhat == Ytest(i)
            hit = hit + 1;
            hittyp(Ctest(i)) = hittyp(Ctest(i)) + 1;
        else
            miss = miss + 1;
            misstyp(Ctest(i)) = misstyp(Ctest(i)) + 1;
        end
    end
    
    acc = hit/(hit+miss);
    typacc = hittyp./(hittyp + misstyp);
end


