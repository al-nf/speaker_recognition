for idx = 0:11
    outdir = sprintf('./specgen/spectrograms/%d', idx);
    mkdir(outdir); % Create a new folder to store spectrogram images
    file = sprintf('./specgen/clips/%d/', idx); % Folder containing audio signals
    file1 = strcat(file, '*.mp3');
    file2 = dir(file1); % Get all mp3 files in the folder
    k = length(file2); % Count the number of mp3 files
    R = 1024; % Window length
    window = hamming(R); % Hamming window
    N = 1024; % Number of FFT points
    L = 512; % Step size
    overlap = R - L; % Overlap of frames
    for i = 1:k
        file3 = strcat(file, file2(i).name); % Absolute path of a single audio file
        file4 = strcat(outdir, '/', file2(i).name, '.jpg'); % Naming and storage of spectrogram images
        if exist(file4, 'file')
            fprintf('Skipping %s, spectrogram already exists.\n', file2(i).name);
            continue;
        end
        [x, fs] = audioread(file3); % Read a single audio file
        x1 = x(:,1); % Take the mono channel
        figure(i);
        specgram(x1, N, fs, window, overlap);
        saveas(gca, file4);
    end
end
    %[B,f,t]=specgram(x(:,1),N,fs,window,overlap);% Second method: specgram returns values, B is amplitude, f is frequency, t is time
    %imagesc(t,f,20*log10(abs(B)));% Plot spectrogram based on returned values (imagesc: plot, parameters are x-axis, y-axis, and values); can also plot other things
    %plot(f,20*log10(abs(B(:,21))))% Frequency domain: B(:,21) corresponds to t[21]=0.23s spectrum
    %plot(t,20*log10(abs(B(21,:))))% Time domain: B(21,:) corresponds to f[21]=816.32Hz waveform
    %colormap(cool);% Change color map
    %axis xy% Note the difference between axis ij and axis xy
    %xlabel('Time/s');% X-axis label
    %ylabel('Frequency/kHz');% Y-axis label
close all;