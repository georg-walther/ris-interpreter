classdef ClassificationLSTM < Classification
    %CLASSIFICATIONLSTM Summary of this class goes here

    methods
        function obj = ClassificationLSTM()
            %CLASSIFICATIONLSTM Construct an instance of this class
            obj@Classification();
        end
        
        function [specs,info] = train(~,trainDocs,YTrain,valDocs,YVal,sequenceLength)
            if nargin < 4
                valDocs = [];
                YVal = [];
                XValidation = [];
                validate = false;
                disp('Training without validation ...');
            elseif nargin < 5
                error('Validation data is provided but the labels (next parameter) are missing!');
            else
                YVal = categorical(YVal);
                validate = true;
                disp('Training with validation ...');
            end
            if nargin < 6
                Stats.plotDocLength(trainDocs);
                disp('Please select a cut off value on the x axis ...');
                [sequenceLength,~] = ginput(1);
                close(gcf());
                sequenceLength = round(sequenceLength);
                disp('Sequence length cut off: ' + string(sequenceLength));
            end
            YTrain = categorical(YTrain);

            enc = wordEncoding(trainDocs);
            % Most of the training documents have fewer than x tokens. Use this as your target length for truncation and padding.
            % Convert the documents to sequences of numeric indices using doc2sequence. To truncate or left-pad the sequences to have length x, set the 'Length' option to x.
            XTrain = doc2sequence(enc,trainDocs,'Length',sequenceLength);
            
            % Define the LSTM network architecture.
            % To input sequence data into the network, include a sequence input layer and set the input size to 1.
            % Next, include a word embedding layer of dimension 50 and the same number of words as the word encoding.
            % Next, include an LSTM layer and set the number of hidden units to 80.
            % To use the LSTM layer for a sequence-to-label classification problem, set the output mode to 'last'.
            % Finally, add a fully connected layer with the same size as the number of classes, a softmax layer, and a classification layer.
            inputSize = 1;
            embeddingDimension = 50;
            numHiddenUnits = 80;
            
            numWords = enc.NumWords;
            numClasses = numel(categories(YTrain));
            
            layers = [ ...
                sequenceInputLayer(inputSize)
                wordEmbeddingLayer(embeddingDimension,numWords)
                lstmLayer(numHiddenUnits,'OutputMode','last')
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
            
            if validate
                XValidation = doc2sequence(enc,valDocs,'Length',sequenceLength);
                validationData = {XValidation,YVal};
            else
                validationData = [];
            end

            options = trainingOptions('adam', ...
                'MiniBatchSize',16, ...
                'MaxEpochs',15, ...
                'GradientThreshold',2, ...
                'Shuffle','every-epoch', ...
                'ValidationData',validationData, ...
                'Plots','training-progress', ...
                'Verbose',false);
            
            [net,info] = trainNetwork(XTrain,YTrain,layers,options);
            specs.net = net;
            specs.enc = enc;
            specs.seqLen = sequenceLength;
        end

        function [YPred,stats] = predict(~,specs,testDocs,YTest)        
            if nargin < 4
                YTest = [];
            end

            XTest = doc2sequence(specs.enc,testDocs,'Length',specs.seqLen);
            YPred = classify(specs.net,XTest);
            
            if ~isempty(YTest)
                stats.acc = sum(string(cell2mat(YPred)) == string(cell2mat(testLabels)))/numel(testLabels);
                rocObj = rocmetrics(testLabels,postProbs,model.ClassNames);
                stats.auc = rocObj.AUC;
                if length(model.ClassNames) == 2
                    confmat = confusionmat(str2double(testLabels),str2double(YPred)); % using strings instead of numbers orders the matrix unpredictably
                    TN = confmat(2, 2);
                    TP = confmat(1, 1);
                    FN = confmat(1, 2);
                    FP = confmat(2, 1);
                    % stats.accuracy = (TP + TN) / (TP + TN + FP + FN);
                    stats.sens = TP / (FN + TP);
                    stats.spec = TN / (TN + FP);
                end
            else
                stats = [];
            end
        end

    end
end

