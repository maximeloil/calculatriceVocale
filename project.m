clear all;
close all;
%musique = wavread('C:\Users\eleves\Documents\Emilien&Maxime\soundlist\maxime\un.wav');
musique =  audioread(['C:\Users\eleves\Documents\Emilien&Maxime\data\wav\0-2.wav']);
Fe  = 44000;
tailleEchan = 64;
N = 2*tailleEchan;
gain = 800;
col1 = musique(:,1);
%col2 = musique(:,2);

longMusique = length(musique);


%ne prend pas en compte la superposition

nombreEchan = fix(longMusique/tailleEchan); % a 20ms

%nombreEchan = 4096;


spectrogramme = zeros(nombreEchan,N/2);



for k=1:nombreEchan
    sample = musique((k-1)*tailleEchan+1:k*tailleEchan,:);
    for t = 1:tailleEchan
        sample(t,:)= sample(t,:) * (0.54-0.46*cos(2*pi*t/tailleEchan));
    end
    a = fft(sample(:,1),N);
    %figure;
    %plot(1:length(a),a);
    tailleA = size(a,1);
    freqs = a(1:tailleA/2);
    spectrogramme(k,:) = freqs;
end
    
%soundsc(col1,Fe);

res = gain*abs(spectrogramme);
res=res';
res2=zeros(size(res,1),size(res,2));
for i=1:size(res,1)
    for j=1:size(res,2)
        res2(i,j)=res(size(res,1)-i+1,j);
    end
end
image(res2);
colormap(jet);