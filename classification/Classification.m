classdef Classification
    %CLASSIFICATION Summary of this class goes here
    
    properties
    end
    
    methods
        function obj = Classification()
            %CLASSIFICATION Construct an instance of this class
            % Detailed explanation goes here
        end
        
        function documents = preprocessText(obj,textData,language) 
            % Tokenize.
            documents = tokenizedDocument(textData);
            
            % Remove tokens containing digits.
            pat = textBoundary + wildcardPattern + digitsPattern + wildcardPattern + textBoundary;
            documents = replace(documents,pat,"");
            
            % Convert to lowercase.
            documents = lower(documents);
            
            % Remove short words.
            documents = removeShortWords(documents,2);
            
            if ismember(lower(language),["english","japanese","german","korean"])
                % Remove stop words.
                documents = removeStopWords(documents);
            end
        end

        function [XTrain,YTrain,XVal,YVal,XTest,YTest] = splitTrainValTest(obj,data,percTrain,percVal)
            %METHOD1 Summary of this method goes here
            % Partition the data into a training partition and a held-out test set. Specify the holdout percentage to be 10%.
            if nargin < 2
                percTrain = 0.9;
            end
            if nargin < 3
                percVal = 1 - percTrain;
            end
            
            cvp = cvpartition(data.Y,'Holdout',percVal);
            dataTrain = data(cvp.training,:);
            dataVal = data(cvp.test,:);

            percTest = 1 - (percTrain + percVal);
            if percTest > 0
                cvp = cvpartition(dataVal.Y,'Holdout',percTest);
                dataVal = data(cvp.training,:);
                dataTest = data(cvp.test,:);
                XTest = dataTest.X;
                YTest = dataTest.Y;
            elseif percTest < 0
                error('Sum of training percentage and validation percentage must be <= 1.');
            else
                XTest = [];
                YTest = [];
            end

            % Extract the text data and labels from the tables.
            XTrain = dataTrain.X;
            XVal = dataVal.X;
            YTrain = dataTrain.Y;
            YVal = dataVal.Y;
        end

        function [features,bags] = extractFeatures(obj,documents,useBERT)
            if nargin < 3
                useBERT = true;
            end

            disp("Extracting features ...");
            minCount = 2;
            wordBag = bagOfWords(documents);
            % Remove words from the bag-of-words model that do not appear more than two times in total. Remove any documents containing no words from the bag-of-words model, and remove the corresponding entries in labels.
            wordBag = removeInfrequentWords(wordBag,minCount);

            bigramBag = bagOfNgrams(documents,'NgramLengths',2);
            bigramBag = removeInfrequentNgrams(bigramBag,minCount,'NgramLengths',2);

            trigramBag = bagOfNgrams(documents,'NgramLengths',3);
            trigramBag = removeInfrequentNgrams(trigramBag,minCount,'NgramLengths',3);

            bags.word = wordBag;
            bags.bigram = bigramBag;
            bags.trigram = trigramBag;

            wordTFIDF = tfidf(wordBag);
            bigramTFIDF = tfidf(bigramBag);
            trigramTFIDF = tfidf(trigramBag);

            features = [wordBag.Counts,bigramBag.Counts,trigramBag.Counts,wordTFIDF,bigramTFIDF,trigramTFIDF];
        end

        function idxBestFeatures = selectFeatures(obj,features,labels)
            disp("Selecting features ...");
            % Feature selection
            [samples,numFeatures] = size(features);
            shuffled = full(features);
            n = 2000;
            if numFeatures > n
                shuffled = shuffled(:, randperm(numFeatures));
                shuffled = shuffled(:,1:n);
            end
            [rho,pval] = corr(shuffled,shuffled);
            halfRho = tril(rho,-1);
            pairwaisePearson = sum(halfRho,'all','omitnan') / ((n * (n - 1) / 2) - nnz(isnan(halfRho)));
            % https://datascience.stackexchange.com/questions/11390/any-rules-of-thumb-on-number-of-features-versus-number-of-instances-small-da
            optNumFeatures = round(Utils.interpolate(abs(pairwaisePearson),0,1,samples-1,sqrt(samples)));
            
            [idx,scores] = fscchi2(full(features),labels);
            idxBestFeatures = idx(1:optNumFeatures);
        end

        function model = train(obj,features,labels)
            disp("Training classifier with hyperparameter optimization ...");
            % rng(5); % For reproducibility
            cvp = cvpartition(labels,'KFold',5,'Stratify',true);
            t = templateTree('Reproducible',true);
            if issparse(features)
                features = full(features); % matlabs hyperparameter optimizations bugs with sparse matrices
            end
            model = fitcensemble(features,labels,'OptimizeHyperparameters','auto','Learners',t, ...
                                 'HyperparameterOptimizationOptions',struct('CVPartition',cvp,'AcquisitionFunctionName','expected-improvement-plus'));
            
            %subspaceModel = fitcensemble(features,labels,'Method','Subspace','CrossVal','on','CVPartition',cvp,'OptimizeHyperparameters','all');
            %adaboostModel = fitcensemble(features,labels,'AdaBoostM2','Subspace','CrossVal','on','CVPartition',cvp,'OptimizeHyperparameters','all');
            %lpboostModel = fitcensemble(features,labels,'LPBoost','Subspace','CrossVal','on','CVPartition',cvp,'OptimizeHyperparameters','all');

            %model = fitcecoc(features,labels,'Learners','linear');

            cvMdl = crossval(model,'CVPartition',cvp);
            cvtrainError = kfoldLoss(cvMdl);
            cvtrainAccuracy = 1-cvtrainError;
            disp(cvtrainAccuracy);
        end

        function acc = test(~,testDocs,testLabels,specs)
            bags = specs.bags;
            model = specs.model;
            idxBest = specs.idxBest;

            % testDocs = FeatureExtraction.preprocessText(XTest);
            wordCounts = encode(bags.word,testDocs);
            bigramCounts = encode(bags.bigram,testDocs);
            trigramCounts = encode(bags.trigram,testDocs);

            wordTFIDF = tfidf(bags.word,testDocs);
            bigramTFIDF = tfidf(bags.bigram,testDocs);
            trigramTFIDF = tfidf(bags.trigram,testDocs);

            features = [wordCounts,bigramCounts,trigramCounts,wordTFIDF,bigramTFIDF,trigramTFIDF];
            features = features(:,idxBest);

            % Predict the labels of the test data using the trained model and calculate the classification accuracy.
            YPred = predict(model,features);
            acc = sum(string(cell2mat(YPred)) == string(cell2mat(testLabels)))/numel(testLabels);
        end

        function net = trainLSTM(obj,trainDocs,YTrain,valDocs,YVal, ...
                                 enc,sequenceLength)
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

    end
end

