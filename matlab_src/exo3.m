close all
N = 10000;
n = 0:N-1;
Fe = 8000;
f1 = 500;
f2 = 5/4*f1;
f3 = 3/2*f1;

t = n/8000;
s1 = cos(2*pi*f1*t);
ind = 1:200;
figure;
f2=490;
s2 = cos(2*pi*f2*t-pi);
% s3 = cos(2*pi*f3*t);
plot(t(ind),s1(ind)+s2(ind),'-');


soundsc(s1+s2,Fe);

%%

nfft = 4096;
s1fft = fft(s1,nfft);
plot(abs(s1fft))