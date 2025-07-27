for idx = 0:11
    path_train = sprintf('./spectrograms/real/%d', idx);
    imds = imageDatastore(path_train, 'IncludeSubfolders', true, 'FileExtensions', {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.mp3'});
    % Limit to at most 150 spectrograms
    if numel(imds.Files) > 150
        idx_sub = randperm(numel(imds.Files), 150);
        imds.Files = imds.Files(idx_sub);
        if ~isempty(imds.Labels)
            imds.Labels = imds.Labels(idx_sub);
        end
    end
    augmenter = imageDataAugmenter('RandXReflection', true);
    augimds = augmentedImageDatastore([64 64], imds, 'DataAugmentation', augmenter);
    filterSize = 5;
    numFilters = 64;
    numLatentInputs = 100;
    projectionSize = [4 4 512];
    layersGenerator = [
        imageInputLayer([1 1 numLatentInputs], 'Normalization', 'none', 'Name', 'in')
        projectAndReshapeLayer(projectionSize, numLatentInputs, 'proj');
        transposedConv2dLayer(filterSize, 4 * numFilters, 'Name', 'tconv1')
        batchNormalizationLayer('Name', 'bnorm1')
        reluLayer('Name', 'relu1')
        transposedConv2dLayer(filterSize, 2 * numFilters, 'Stride', 2, 'Cropping', 'same', 'Name', 'tconv2')
        batchNormalizationLayer('Name', 'bnorm2')
        reluLayer('Name', 'relu2')
        transposedConv2dLayer(filterSize, numFilters, 'Stride', 2, 'Cropping', 'same', 'Name', 'tconv3')
        batchNormalizationLayer('Name', 'bnorm3')
        reluLayer('Name', 'relu3')
        transposedConv2dLayer(filterSize, 3, 'Stride', 2, 'Cropping', 'same', 'Name', 'tconv4')
        tanhLayer('Name', 'tanh')];
    lgraphGenerator = layerGraph(layersGenerator);
    dlnetGenerator = dlnetwork(lgraphGenerator);
    dropoutProb = 0.5;
    numFilters = 64;
    scale = 0.2;
    inputSize = [64 64 3];
    filterSize = 5;
    layersDiscriminator = [
        imageInputLayer(inputSize, 'Normalization', 'none', 'name', 'in')
        dropoutLayer(0.5, 'name', 'dropout')
        convolution2dLayer(filterSize, numFilters, 'Stride', 2, 'Padding', 'same', 'Name', 'conv1')
        leakyReluLayer(scale, 'name', 'lrelu1')
        convolution2dLayer(filterSize, 2 * numFilters, 'Stride', 2, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        leakyReluLayer(scale, 'Name', 'lrelu2')
        convolution2dLayer(filterSize, 4 * numFilters, 'Stride', 2, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        leakyReluLayer(scale, 'Name', 'lrelu3')
        convolution2dLayer(filterSize, 8 * numFilters, 'Stride', 2, 'Padding', 'same', 'Name', 'conv4')
        batchNormalizationLayer('Name', 'bn4')
        leakyReluLayer(scale, 'Name', 'lrelu4')
        convolution2dLayer(4, 1, 'Name', 'conv5')];
    lgraphDiscriminator = layerGraph(layersDiscriminator);
    dlnetDiscriminator = dlnetwork(lgraphDiscriminator);
    numEpochs = 160;
    miniBatchSize = 3;
    learnRate = 0.0001;
    gradientDecayFactor = 0.5;
    squaredGradientDecayFactor = 0.999;
    flipFactor = 0.3;
    validationFrequency = 20;
    augimds.MiniBatchSize = miniBatchSize;

    executionEnvironment = "auto";

    mbq = minibatchqueue(augimds, ...
        'MiniBatchSize', miniBatchSize, ...
        'PartialMiniBatch', 'discard', ...
        'MiniBatchFcn', @preprocessMiniBatch, ...
        'MiniBatchFormat', 'SSCB', ...
        'OutputEnvironment', executionEnvironment);
    trailingAvgGenerator = [];
    trailingAvgSqGenerator = [];
    trailingAvgDiscriminator = [];
    trailingAvgSqDiscriminator = [];
    numValidationImages = 40;
    ZValidation = randn(1, 1, numLatentInputs, numValidationImages, 'single');
    dlZValidation = dlarray(ZValidation, 'SSCB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlZValidation = gpuArray(dlZValidation);
    end

    % Create directory for saving generated spectrograms
    outputDir = sprintf('./spectrograms/gen/%d', idx);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    f = figure;
    f.Position(3) = 2 * f.Position(3);
    imageAxes = subplot(1, 2, 1);
    scoreAxes = subplot(1, 2, 2);
    lineScoreGenerator = animatedline(scoreAxes, 'Color', [0 0.447 0.741]);
    lineScoreDiscriminator = animatedline(scoreAxes, 'Color', [0.85 0.325 0.098]);
    legend('Generator', 'Discriminator');
    ylim([0 1])
    xlabel("Iteration")
    ylabel("Score")
    grid on

    iteration = 0;
    start = tic;

    for epoch = 1:numEpochs
        shuffle(mbq);
        while hasdata(mbq)
            iteration = iteration + 1;
            dlX = next(mbq);
            Z = randn(1, 1, numLatentInputs, size(dlX, 4), 'single');
            dlZ = dlarray(Z, 'SSCB');
            if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
                dlZ = gpuArray(dlZ);
            end
            [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
                dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor);
            dlnetGenerator.State = stateGenerator;
            [dlnetDiscriminator, trailingAvgDiscriminator, trailingAvgSqDiscriminator] = ...
                adamupdate(dlnetDiscriminator, gradientsDiscriminator, ...
                trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
                learnRate, gradientDecayFactor, squaredGradientDecayFactor);
            [dlnetGenerator, trailingAvgGenerator, trailingAvgSqGenerator] = ...
                adamupdate(dlnetGenerator, gradientsGenerator, ...
                trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
                learnRate, gradientDecayFactor, squaredGradientDecayFactor);
            if mod(iteration, validationFrequency) == 0 || iteration == 1
                dlXGeneratedValidation = predict(dlnetGenerator, dlZValidation);
                I = imtile(extractdata(dlXGeneratedValidation));
                I = rescale(I);
                subplot(1, 2, 1);
                image(imageAxes, I)
                xticklabels([]);
                yticklabels([]);
                title("Generated Images");
                if epoch > 80
                    outFile = fullfile(outputDir, sprintf('generated_iter_%06d.png', iteration));
                    imwrite(I, outFile);
                end
            end
            subplot(1, 2, 2)
            addpoints(lineScoreGenerator, iteration, double(gather(extractdata(scoreGenerator))));
            addpoints(lineScoreDiscriminator, iteration, double(gather(extractdata(scoreDiscriminator))));
            D = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
            title("Epoch: " + epoch + ", " + "Iteration: " + iteration + ", " + "Elapsed: " + string(D))
            drawnow
        end
    end
    close(f);
end
