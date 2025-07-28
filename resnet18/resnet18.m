% Load pretrained ResNet18 parameters
params = load("./resnet18/params_2024_11_09__14_08_44.mat");

% Directories for real and GAN spectrograms
realDir = './spectrograms/real';
genDir = './spectrograms/gen';


% Load image datastores, both with speaker labels from folder names
imdsReal = imageDatastore(realDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsGen = imageDatastore(genDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Mixing proportions
numTotal = min(numel(imdsReal.Files) + numel(imdsGen.Files), 1000); % Use up to 1000 images
genPercents = 5:5:40; % 5%, 10%, ..., 40%
realPercents = 100 - genPercents; % 95%, 90%, ..., 60%
numEpochs = numel(genPercents);

inputSize = [224 224 3];

numRuns = 5;
for runIdx = 1:numRuns
    for epochIdx = 1:numEpochs
        fprintf('Starting run %d/%d, epoch %d/%d...\n', runIdx, numRuns, epochIdx, numEpochs);
    % Build the layer graph for each run
    lgraph = layerGraph();
    
    %% Add layer branches
    % Add network branches to the layer graph. Each branch is a linear group of layers.

    tempLayers = [
        imageInputLayer([224 224 3],"Name","data","Normalization","zscore","Mean",params.data.Mean,"StandardDeviation",params.data.StandardDeviation)
        convolution2dLayer([7 7],64,"Name","conv1","BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2],"Bias",params.conv1.Bias,"Weights",params.conv1.Weights)
        batchNormalizationLayer("Name","bn_conv1","Offset",params.bn_conv1.Offset,"Scale",params.bn_conv1.Scale,"TrainedMean",params.bn_conv1.TrainedMean,"TrainedVariance",params.bn_conv1.TrainedVariance)
        reluLayer("Name","conv1_relu")
        maxPooling2dLayer([3 3],"Name","pool1","Padding",[1 1 1 1],"Stride",[2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2a_branch2a.Bias,"Weights",params.res2a_branch2a.Weights)
        batchNormalizationLayer("Name","bn2a_branch2a","Offset",params.bn2a_branch2a.Offset,"Scale",params.bn2a_branch2a.Scale,"TrainedMean",params.bn2a_branch2a.TrainedMean,"TrainedVariance",params.bn2a_branch2a.TrainedVariance)
        reluLayer("Name","res2a_branch2a_relu")
        convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2a_branch2b.Bias,"Weights",params.res2a_branch2b.Weights)
        batchNormalizationLayer("Name","bn2a_branch2b","Offset",params.bn2a_branch2b.Offset,"Scale",params.bn2a_branch2b.Scale,"TrainedMean",params.bn2a_branch2b.TrainedMean,"TrainedVariance",params.bn2a_branch2b.TrainedVariance)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","res2a")
        reluLayer("Name","res2a_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],64,"Name","res2b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2b_branch2a.Bias,"Weights",params.res2b_branch2a.Weights)
        batchNormalizationLayer("Name","bn2b_branch2a","Offset",params.bn2b_branch2a.Offset,"Scale",params.bn2b_branch2a.Scale,"TrainedMean",params.bn2b_branch2a.TrainedMean,"TrainedVariance",params.bn2b_branch2a.TrainedVariance)
        reluLayer("Name","res2b_branch2a_relu")
        convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2b_branch2b.Bias,"Weights",params.res2b_branch2b.Weights)
        batchNormalizationLayer("Name","bn2b_branch2b","Offset",params.bn2b_branch2b.Offset,"Scale",params.bn2b_branch2b.Scale,"TrainedMean",params.bn2b_branch2b.TrainedMean,"TrainedVariance",params.bn2b_branch2b.TrainedVariance)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","res2b")
        reluLayer("Name","res2b_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2],"Bias",params.res3a_branch2a.Bias,"Weights",params.res3a_branch2a.Weights)
        batchNormalizationLayer("Name","bn3a_branch2a","Offset",params.bn3a_branch2a.Offset,"Scale",params.bn3a_branch2a.Scale,"TrainedMean",params.bn3a_branch2a.TrainedMean,"TrainedVariance",params.bn3a_branch2a.TrainedVariance)
        reluLayer("Name","res3a_branch2a_relu")
        convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res3a_branch2b.Bias,"Weights",params.res3a_branch2b.Weights)
        batchNormalizationLayer("Name","bn3a_branch2b","Offset",params.bn3a_branch2b.Offset,"Scale",params.bn3a_branch2b.Scale,"TrainedMean",params.bn3a_branch2b.TrainedMean,"TrainedVariance",params.bn3a_branch2b.TrainedVariance)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],128,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res3a_branch1.Bias,"Weights",params.res3a_branch1.Weights)
        batchNormalizationLayer("Name","bn3a_branch1","Offset",params.bn3a_branch1.Offset,"Scale",params.bn3a_branch1.Scale,"TrainedMean",params.bn3a_branch1.TrainedMean,"TrainedVariance",params.bn3a_branch1.TrainedVariance)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","res3a")
        reluLayer("Name","res3a_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],128,"Name","res3b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res3b_branch2a.Bias,"Weights",params.res3b_branch2a.Weights)
        batchNormalizationLayer("Name","bn3b_branch2a","Offset",params.bn3b_branch2a.Offset,"Scale",params.bn3b_branch2a.Scale,"TrainedMean",params.bn3b_branch2a.TrainedMean,"TrainedVariance",params.bn3b_branch2a.TrainedVariance)
        reluLayer("Name","res3b_branch2a_relu")
        convolution2dLayer([3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res3b_branch2b.Bias,"Weights",params.res3b_branch2b.Weights)
        batchNormalizationLayer("Name","bn3b_branch2b","Offset",params.bn3b_branch2b.Offset,"Scale",params.bn3b_branch2b.Scale,"TrainedMean",params.bn3b_branch2b.TrainedMean,"TrainedVariance",params.bn3b_branch2b.TrainedVariance)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","res3b")
        reluLayer("Name","res3b_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2],"Bias",params.res4a_branch2a.Bias,"Weights",params.res4a_branch2a.Weights)
        batchNormalizationLayer("Name","bn4a_branch2a","Offset",params.bn4a_branch2a.Offset,"Scale",params.bn4a_branch2a.Scale,"TrainedMean",params.bn4a_branch2a.TrainedMean,"TrainedVariance",params.bn4a_branch2a.TrainedVariance)
        reluLayer("Name","res4a_branch2a_relu")
        convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res4a_branch2b.Bias,"Weights",params.res4a_branch2b.Weights)
        batchNormalizationLayer("Name","bn4a_branch2b","Offset",params.bn4a_branch2b.Offset,"Scale",params.bn4a_branch2b.Scale,"TrainedMean",params.bn4a_branch2b.TrainedMean,"TrainedVariance",params.bn4a_branch2b.TrainedVariance)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],256,"Name","res4a_branch1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res4a_branch1.Bias,"Weights",params.res4a_branch1.Weights)
        batchNormalizationLayer("Name","bn4a_branch1","Offset",params.bn4a_branch1.Offset,"Scale",params.bn4a_branch1.Scale,"TrainedMean",params.bn4a_branch1.TrainedMean,"TrainedVariance",params.bn4a_branch1.TrainedVariance)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","res4a")
        reluLayer("Name","res4a_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],256,"Name","res4b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res4b_branch2a.Bias,"Weights",params.res4b_branch2a.Weights)
        batchNormalizationLayer("Name","bn4b_branch2a","Offset",params.bn4b_branch2a.Offset,"Scale",params.bn4b_branch2a.Scale,"TrainedMean",params.bn4b_branch2a.TrainedMean,"TrainedVariance",params.bn4b_branch2a.TrainedVariance)
        reluLayer("Name","res4b_branch2a_relu")
        convolution2dLayer([3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res4b_branch2b.Bias,"Weights",params.res4b_branch2b.Weights)
        batchNormalizationLayer("Name","bn4b_branch2b","Offset",params.bn4b_branch2b.Offset,"Scale",params.bn4b_branch2b.Scale,"TrainedMean",params.bn4b_branch2b.TrainedMean,"TrainedVariance",params.bn4b_branch2b.TrainedVariance)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","res4b")
        reluLayer("Name","res4b_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2],"Bias",params.res5a_branch2a.Bias,"Weights",params.res5a_branch2a.Weights)
        batchNormalizationLayer("Name","bn5a_branch2a","Offset",params.bn5a_branch2a.Offset,"Scale",params.bn5a_branch2a.Scale,"TrainedMean",params.bn5a_branch2a.TrainedMean,"TrainedVariance",params.bn5a_branch2a.TrainedVariance)
        reluLayer("Name","res5a_branch2a_relu")
        convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res5a_branch2b.Bias,"Weights",params.res5a_branch2b.Weights)
        batchNormalizationLayer("Name","bn5a_branch2b","Offset",params.bn5a_branch2b.Offset,"Scale",params.bn5a_branch2b.Scale,"TrainedMean",params.bn5a_branch2b.TrainedMean,"TrainedVariance",params.bn5a_branch2b.TrainedVariance)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],512,"Name","res5a_branch1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res5a_branch1.Bias,"Weights",params.res5a_branch1.Weights)
        batchNormalizationLayer("Name","bn5a_branch1","Offset",params.bn5a_branch1.Offset,"Scale",params.bn5a_branch1.Scale,"TrainedMean",params.bn5a_branch1.TrainedMean,"TrainedVariance",params.bn5a_branch1.TrainedVariance)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","res5a")
        reluLayer("Name","res5a_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],512,"Name","res5b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res5b_branch2a.Bias,"Weights",params.res5b_branch2a.Weights)
        batchNormalizationLayer("Name","bn5b_branch2a","Offset",params.bn5b_branch2a.Offset,"Scale",params.bn5b_branch2a.Scale,"TrainedMean",params.bn5b_branch2a.TrainedMean,"TrainedVariance",params.bn5b_branch2a.TrainedVariance)
        reluLayer("Name","res5b_branch2a_relu")
        convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res5b_branch2b.Bias,"Weights",params.res5b_branch2b.Weights)
        batchNormalizationLayer("Name","bn5b_branch2b","Offset",params.bn5b_branch2b.Offset,"Scale",params.bn5b_branch2b.Scale,"TrainedMean",params.bn5b_branch2b.TrainedMean,"TrainedVariance",params.bn5b_branch2b.TrainedVariance)];
    lgraph = addLayers(lgraph,tempLayers);

    % Set number of classes to number of speakers
    speakerClasses = categories(imdsReal.Labels);
    numSpeakers = numel(speakerClasses);
    tempLayers = [
        additionLayer(2,"Name","res5b")
        reluLayer("Name","res5b_relu")
        globalAveragePooling2dLayer("Name","pool5")
        fullyConnectedLayer(numSpeakers,"Name","fc_final")
        softmaxLayer("Name","prob")
        classificationLayer("Name","ClassificationLayer_predictions","Classes",categorical(speakerClasses))];
    lgraph = addLayers(lgraph,tempLayers);

    % Clear temporary variables
    clear tempLayers;

    %% Connect layer branches
    % Connect all branches of the network to create the network graph.

    lgraph = connectLayers(lgraph,"pool1","res2a_branch2a");
    lgraph = connectLayers(lgraph,"pool1","res2a/in2");
    lgraph = connectLayers(lgraph,"bn2a_branch2b","res2a/in1");
    lgraph = connectLayers(lgraph,"res2a_relu","res2b_branch2a");
    lgraph = connectLayers(lgraph,"res2a_relu","res2b/in2");
    lgraph = connectLayers(lgraph,"bn2b_branch2b","res2b/in1");
    lgraph = connectLayers(lgraph,"res2b_relu","res3a_branch2a");
    lgraph = connectLayers(lgraph,"res2b_relu","res3a_branch1");
    lgraph = connectLayers(lgraph,"bn3a_branch2b","res3a/in1");
    lgraph = connectLayers(lgraph,"bn3a_branch1","res3a/in2");
    lgraph = connectLayers(lgraph,"res3a_relu","res3b_branch2a");
    lgraph = connectLayers(lgraph,"res3a_relu","res3b/in2");
    lgraph = connectLayers(lgraph,"bn3b_branch2b","res3b/in1");
    lgraph = connectLayers(lgraph,"res3b_relu","res4a_branch2a");
    lgraph = connectLayers(lgraph,"res3b_relu","res4a_branch1");
    lgraph = connectLayers(lgraph,"bn4a_branch2b","res4a/in1");
    lgraph = connectLayers(lgraph,"bn4a_branch1","res4a/in2");
    lgraph = connectLayers(lgraph,"res4a_relu","res4b_branch2a");
    lgraph = connectLayers(lgraph,"res4a_relu","res4b/in2");
    lgraph = connectLayers(lgraph,"bn4b_branch2b","res4b/in1");
    lgraph = connectLayers(lgraph,"res4b_relu","res5a_branch2a");
    lgraph = connectLayers(lgraph,"res4b_relu","res5a_branch1");
    lgraph = connectLayers(lgraph,"bn5a_branch2b","res5a/in1");
    lgraph = connectLayers(lgraph,"bn5a_branch1","res5a/in2");
    lgraph = connectLayers(lgraph,"res5a_relu","res5b_branch2a");
    lgraph = connectLayers(lgraph,"res5a_relu","res5b/in2");
    lgraph = connectLayers(lgraph,"bn5b_branch2b","res5b/in1");

    % Mix data for this epoch
    genProp = genPercents(epochIdx) / 100;
    realProp = realPercents(epochIdx) / 100;
    numGen = round(numTotal * genProp);
    numReal = round(numTotal * realProp);
    idxGen = randperm(numel(imdsGen.Files), min(numGen, numel(imdsGen.Files)));
    idxReal = randperm(numel(imdsReal.Files), min(numReal, numel(imdsReal.Files)));
    imdsMix = imageDatastore([imdsGen.Files(idxGen); imdsReal.Files(idxReal)], ...
        'Labels', [imdsGen.Labels(idxGen); imdsReal.Labels(idxReal)]);
    % Split into training/validation
    [imdsTrain, imdsVal] = splitEachLabel(imdsMix, 0.8, 'randomized');
    % Preprocess images
    augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
    augimdsVal = augmentedImageDatastore(inputSize, imdsVal, 'ColorPreprocessing', 'gray2rgb');
    % Set training options
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.001, ...
        'MaxEpochs',10, ...
        'MiniBatchSize',32, ...
        'ValidationData',augimdsVal, ...
        'ValidationFrequency',30, ...
        'Verbose',false);
    % Train the network
    net = trainNetwork(augimdsTrain, lgraph, options);
    % Test accuracy
    preds = classify(net, augimdsVal);
    acc = mean(preds == imdsVal.Labels);
    logMsg = sprintf('Epoch %d: Gen %.0f%%, Real %.0f%%, Accuracy: %.4f\n', epochIdx, genPercents(epochIdx), realPercents(epochIdx), acc);
    fid = fopen('training_log.txt', 'a');
    if fid ~= -1
        fprintf(fid, '%s', logMsg);
        fclose(fid);
    else
        warning('Could not open training_log.txt for writing.');
    end
    fid = fopen('training_log.txt', 'a');
    end
    if fid ~= -1
        fprintf(fid, '\n');
        fclose(fid);
    
        % Save the trained network and accuracy
        save(sprintf('resnet18_speaker_mix_%d_%d_run%d.mat', genPercents(epochIdx), realPercents(epochIdx), runIdx), 'net', 'acc');
    end
end
close all;