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
            
            if percVal == 0 && percTrain == 1
                disp('Skipping data splitting ...');
                dataTrain = data;
                XVal = [];
                YVal = [];
            else
                cvp = cvpartition(data.Y,'Holdout',percVal);
                dataTrain = data(cvp.training,:);
                dataVal = data(cvp.test,:);

                XVal = dataVal.X;
                YVal = dataVal.Y;
            end

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
            YTrain = dataTrain.Y;
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

        function [model,target] = train(obj,features,labels,target,maxEvals)
            if nargin < 4 || isempty(target)
                targetClass = [];
                targetMetric = [];
                targetValue = [];
            else
                targetClass = string(target.class);
                targetMetric = string(target.metric);
                targetValue = double(target.value);
            end
            if nargin < 5 || isempty(maxEvals)
                maxEvals = 10;
            end

            disp("Training classifier with hyperparameter optimization ...");
            if issparse(features)
                features = full(features); % matlabs hyperparameter optimization bugs with sparse matrices
            end

            cvp = cvpartition(labels,'KFold',5,'Stratify',true);
            classes = unique(labels);
            numClasses = length(classes);
            if  numClasses < 3
                disp('Identified as 2 (or 1) class classification.');
                svm = fitcsvm(features,labels, ...
                    'OptimizeHyperparameters','all','HyperparameterOptimizationOptions', ...
                    struct('CVPartition',cvp,'AcquisitionFunctionName','expected-improvement-plus', ...
                    'MaxObjectiveEvaluations',maxEvals,'UseParallel',true));
                svmPost = fitPosterior(svm);
            else
                disp('Identified as multi-class classification.');
                [svmPost,~] = fitcecoc(features,labels,'Learners','svm','FitPosterior',true, ...
                    'OptimizeHyperparameters','all','HyperparameterOptimizationOptions', ...
                    struct('CVPartition',cvp,'AcquisitionFunctionName','expected-improvement-plus', ...
                    'MaxObjectiveEvaluations',maxEvals,'UseParallel',true));
            end
            
            tree = templateTree();
            ensemble = fitcensemble(features,labels,'OptimizeHyperparameters','all','Learners',tree, ...
                                    'HyperparameterOptimizationOptions',struct('CVPartition',cvp,'AcquisitionFunctionName','expected-improvement-plus', ...
                                    'MaxObjectiveEvaluations',maxEvals,'UseParallel',true));

            modelList = {svmPost,ensemble};
            accSVM = evalCV(obj,svmPost,cvp);
            accEnsemble = evalCV(obj,ensemble,cvp);
            [~,idx] = max([accSVM,accEnsemble]);
            if idx == 1
                disp('SVM perfomed best.');
            elseif idx == 2
                disp('Tree ensemble performed best.');
            end
            model = modelList{idx};

            if ~isempty(targetClass) && ~isempty(targetMetric) && ~isempty(targetValue)
                disp('Attempting to find threshold that yields: ' ...
                    + string(targetMetric) + '=' + string(targetValue) + ' for class ' + string(targetClass) + ' ...');
                [~,postProbs] = resubPredict(model);
                rocObj = rocmetrics(labels,postProbs,model.ClassNames);
                plot(rocObj);
                classROC = rocObj.Metrics(strcmp(rocObj.Metrics.ClassName,targetClass),:);
    
                if strcmp(targetMetric,'sens') || strcmp(targetMetric,'sensitivity') || strcmp(targetMetric,'Sensititvity') || strcmp(targetMetric,'TPR') || strcmp(targetMetric,'tpr')
                    metricCol = classROC.TruePositiveRate;
                elseif strcmp(targetMetric,'spec') || strcmp(targetMetric,'specificity') || strcmp(targetMetric,'Specificity') || strcmp(targetMetric,'TNR') || strcmp(targetMetric,'tnr')
                    metricCol = 1 - classROC.FalsePositiveRate;
                else
                    error('Specified targetMetric parameter must be either sensitivity or specitivity (as a string).');
                end
                diffList = abs(metricCol-targetValue);
                [~,i1] = min(diffList);
                diffList(i1) = Inf;
                [~,i2] = min(diffList);
                t1 = classROC.Threshold(i1);
                t2 = classROC.Threshold(i2);
                if metricCol(i1) == metricCol(i2)
                    target.thresh = t1; % interpolation would yield Inf,-Inf,or NaN -> use the threshold that is closest to the targetValue
                else
                    target.thresh = Utils.interpolate(targetValue,metricCol(i1),metricCol(i2),t1,t2);
                end
            else
                target = [];
            end
        end

        function cvtrainAccuracy = evalCV(~,model,cvp)
            cvMdl = crossval(model,'CVPartition',cvp);
            cvtrainError = kfoldLoss(cvMdl);
            cvtrainAccuracy = 1-cvtrainError;
            disp('Mean accuracy over the CV-folds: ' + string(cvtrainAccuracy));
        end

        function [YPred,stats] = predict(~,specs,testDocs,testLabels)
            if nargin < 4
                testLabels = [];
            end
            bags = specs.bags;
            model = specs.model;
            idxBest = specs.idxBest;
            target = specs.target;

            % testDocs = FeatureExtraction.preprocessText(XTest);
            wordCounts = encode(bags.word,testDocs);
            bigramCounts = encode(bags.bigram,testDocs);
            trigramCounts = encode(bags.trigram,testDocs);

            wordTFIDF = tfidf(bags.word,testDocs);
            bigramTFIDF = tfidf(bags.bigram,testDocs);
            trigramTFIDF = tfidf(bags.trigram,testDocs);

            features = [wordCounts,bigramCounts,trigramCounts,wordTFIDF,bigramTFIDF,trigramTFIDF];
            features = features(:,idxBest);

            features = full(features); % from sparse to normal matrix
            % Predict the labels of the test data using the trained model and calculate the classification accuracy.
            [YPred,postProbs] = predict(model,features);
            if ~isempty(target)
                col = strcmp(model.ClassNames,target.class);
                tmpProbs = postProbs;
                tmpProbs(:,col) = 0;
                [~,repCol] = max(tmpProbs,[],2) ;
                YPred = model.ClassNames(repCol);

                classProbs = postProbs(:,col);
                YPred(classProbs >= target.thresh) = {target.class};
            end

            if ~isempty(testLabels)
                stats.acc= sum(string(cell2mat(YPred)) == string(cell2mat(testLabels)))/numel(testLabels);
                rocObj = rocmetrics(testLabels,postProbs,model.ClassNames);
                stats.auc = rocObj.AUC;
                if length(model.ClassNames) == 2
                    confmat = confusionmat(testLabels,YPred);
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

