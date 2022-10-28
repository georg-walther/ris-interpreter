classdef Classification
    %CLASSIFICATION Summary of this class goes here
    
    properties
        data;
    end
    
    methods
        function obj = Classification(data)
            %CLASSIFICATION Construct an instance of this class
            % Detailed explanation goes here
            obj.data = data;
        end
        
        function [XTrain,YTrain,XVal,YVal,XTest,YTest] = splitTrainValTest(obj,percTrain,percVal)
            %METHOD1 Summary of this method goes here
            % Partition the data into a training partition and a held-out test set. Specify the holdout percentage to be 10%.
            if nargin < 2
                percTrain = 0.9;
            end
            if nargin < 3
                percVal = 1 - percTrain;
            end
            
            cvp = cvpartition(obj.data.Label,'Holdout',percVal);
            dataTrain = obj.data(cvp.training,:);
            dataVal = obj.data(cvp.val,:);

            percTest = 1 - (percTrain + percVal);
            if percTest > 0
                cvp = cvpartition(dataVal.Label,'Holdout',percTest);
                dataVal = obj.data(cvp.training,:);
                dataTest = obj.data(cvp.val,:);
            elseif percTest < 0
                error('Sum of training percentage and validation percentage must be <= 1.');
            else
                dataTest = [];
            end

            % Extract the text data and labels from the tables.
            XTrain = dataTrain.Text;
            XVal = dataVal.Text;
            XTest = dataTest.Text;
            YTrain = dataTrain.Label;
            YVal = dataVal.Label;
            YTest = dataTest.Label;
        end

        function model = trainLinearModel(obj)
            model = fitcecoc(XTrain,YTrain,'Learners','linear');
        end

        function net = trainLSTM(obj,trainDocs,YTrain,valDocs,YVal, ...
                           enc,sequenceLength)
            % Most of the training documents have fewer than x tokens. Use this as your target length for truncation and padding.
            % Convert the documents to sequences of numeric indices using doc2sequence. To truncate or left-pad the sequences to have length x, set the 'Length' option to x.
            % TODO: find this cutoff automatically using otsu-thresholding
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
            
            YValidation = YTest;
            documentsValidation = preprocessText(textDataTest); % TODO: change to validation later
            XValidation = doc2sequence(enc,documentsValidation,'Length',sequenceLength);
            
            options = trainingOptions('adam', ...
                'MiniBatchSize',16, ...
                'GradientThreshold',2, ...
                'Shuffle','every-epoch', ...
                'ValidationData',{XValidation,YValidation}, ...
                'Plots','training-progress', ...
                'Verbose',false);
            
            net = trainNetwork(XTrain,YTrain,layers,options);
        end

        function acc = test(obj,model,data,bag,YTest)
            documentsTest = preprocessText(data);
            XTest = encode(bag,documentsTest);
            
            % Predict the labels of the test data using the trained model and calculate the classification accuracy.
            
            YPred = predict(model,XTest);
            acc = sum(YPred == YTest)/numel(YTest);
        end

    end
end

