function ypred = mypredict1(tbl)
%#function fitcensemble
load('trainedModelnew.mat');
ypred = predict(mdT1,tbl);
end