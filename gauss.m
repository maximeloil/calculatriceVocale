close all;
clear all;
sig = 1;
mu = 0;


X = [-10:0.01:10];


Y = 1./(sig*sqrt(2*pi)) .* exp(-((X-mu).^2 / (2*(sig^2))));
plot(X,Y);
hold on;
Bruit = 0.05*randn(1,length(X));

Z = Y + Bruit;
plot(X,Z,'r');

nEch = 100;

nPts = length(Z);

res = zeros(1,nPts);
for i =(floor(nEch/2)+1):(nPts-floor(nEch/2))
    res(i) = mean(Z(i-floor(nEch/2):i+floor(nEch/2)));
end

plot(X,res,'g');
    


