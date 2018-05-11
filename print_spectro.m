close all
for person = 2:5
    figure;
    for k = 0:3
        music = audioread(['C:\Users\eleves\Documents\Emilien&Maxime\data\wav\' num2str(k) '-' num2str(person) '.wav']);
        l = size(music);
        if l(2) == 2
            music = music(:,1);
        end
        subplot(2,2,k+1)
        spectrogram(music,hamming(floor(length(music)/100)));
        %plot(music);
        title(num2str(k));
        colormap jet;
        soundsc(music,44100);
        pause(1);
    end
end
    