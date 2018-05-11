clear all;
close all;

N = 44000;


son1 = wavread('C:\Users\eleves\Documents\Emilien&Maxime\soundlist\emilien\deux.wav');

son1 = son1(:,1);
sonfft1 = abs(fft(son1,N));
sonfft1 = sonfft1(1:N/2);


ecarts_quad = zeros(13);
for k=0:13
    son = wavread(['C:\Users\eleves\Documents\Emilien&Maxime\soundlist\maxime\',int2str(k),'.wav']);
    son = son(:,1);
    sonfft = abs(fft(son,N));
    sonfft = sonfft(1:N/2);
    ecarts_quad(k) = sum((sonfft1-sonfft).^2);
end



figure;
l = size(sonfft1,1);
plot(1:l, sonfft1);
%soundsc(col1,Fe);

[val,ind] = min(ecarts_quad);

