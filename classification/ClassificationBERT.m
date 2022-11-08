classdef ClassificationBERT < Classification
    %CLASSFICATIONBERT Summary of this class goes here
    %   Detailed explanation goes here
    
    methods
        function obj = ClassificationBERT()
            %CLASSFICATIONBERT Construct an instance of this class
            obj@Classification();
        end
        
        function [tokens,mdl] = extractTokens(obj,txtArray,toEnglish)
            if nargin < 3
                toEnglish = true;
            end
            if toEnglish
                path = append('.',filesep,'data',filesep,'translatedTable.csv');
                if isfile(path)
                    txtArray = readmatrix(path,delimitedTextImportOptions('Delimiter',',','VariableTypes','string'));
                    disp('Loaded ' + string(path) + ' as translation.');
                else
                    % Translate to English
                    parfor i = 1:size(txtArray,1)
                        disp('Translating to English ...');
                        disp('Iterating row ' + string(i));
                        engText = translate(txtArray(i,:));
                        txtArray(i,:) = engText;
                    end
                    writematrix(txtArray,path,'Delimiter',',');
                end
            end
                
            disp('Extracting tokens ...');
            % Load a pretrained BERT model using the |bert| function. The model
            % consists of a tokenizer that encodes text as sequences of integers, and
            % a structure of parameters.
            mdl = bert();
            
            % View the BERT model tokenizer. The tokenizer encodes text as sequences of
            % integers and holds the details of padding, start, separator and mask
            % tokens.
            tokenizer = mdl.Tokenizer;
            
            % Encode the text data using the BERT model tokenizer using the |encode|
            % function and add the tokens to the training data table.
            tokens = encode(tokenizer,txtArray);
        end

        function net = train(obj,mdl,tokensTrain,tokensVal,YTrain,YVal)
            disp('Preparing data for training ...');
            YTrain = categorical(YTrain);
            YVal = categorical(YVal);
            % Convert the documents to feature vectors using the BERT model as a
            % feature extractor.
            
            % To extract the features of the training data by iterating over
            % mini-batches, create a |minibatchqueue| object.
            
            % Mini-batch queues require a single datastore that outputs both the
            % predictors and responses. Create array datastores containing the training
            % BERT tokens and labels and combine them using the |combine| function.
            dsXTrain = arrayDatastore(tokensTrain,"OutputType","same");
            dsYTrain = arrayDatastore(YTrain);
            cdsTrain = combine(dsXTrain,dsYTrain);
            
            % Create a combined datastore for the validation data using the same steps.
            dsXVal = arrayDatastore(tokensVal,"OutputType","same");
            dsYVal = arrayDatastore(YVal);
            cdsVal = combine(dsXVal,dsYVal);
            
            %%
            % Create a mini-batch queue for the training data. Specify a mini-batch
            % size of 32 and preprocess the mini-batches using the
            % |preprocessPredictors| function, listed at the end of the example.
            miniBatchSize = 32;
            paddingValue = mdl.Tokenizer.PaddingCode;
            maxSequenceLength = mdl.Parameters.Hyperparameters.NumContext;
            
            mbqTrain = minibatchqueue(cdsTrain,1,...
                "MiniBatchSize",miniBatchSize, ...
                "MiniBatchFcn",@(X) preprocessPredictors(obj,X,paddingValue,maxSequenceLength));
            
            %%%
            % Create a mini-batch queue for the validation data using the same steps.
            mbqValidation = minibatchqueue(cdsVal,1,...
                "MiniBatchSize",miniBatchSize, ...
                "MiniBatchFcn",@(X) preprocessPredictors(obj,X,paddingValue,maxSequenceLength));
            
            %%
            % To speed up feature extraction. Convert the BERT model weights to
            % gpuArray if a GPU is available.
            if canUseGPU
                mdl.Parameters.Weights = dlupdate(@gpuArray,mdl.Parameters.Weights);
            end
            
            %%
            % Convert the training sequences of BERT model tokens to a
            % |N|-by-|embeddingDimension| array of feature vectors, where |N| is the
            % number of training observations and |embeddingDimension| is the dimension
            % of the BERT embedding.
            
            featuresTrain = [];
            reset(mbqTrain);
            while hasdata(mbqTrain)
                X = next(mbqTrain);
                features = bertEmbed(obj,X,mdl.Parameters);
                featuresTrain = [featuresTrain gather(extractdata(features))];
            end
            
            %%
            % Transpose the training data to have size |N|-by-|embeddingDimension|.
            featuresTrain = featuresTrain.';
            
            %%
            % Convert the validation data to feature vectors using the same steps.
            featuresValidation = [];
            
            reset(mbqValidation);
            while hasdata(mbqValidation)
                X = next(mbqValidation);
                features = bertEmbed(obj,X,mdl.Parameters);
                featuresValidation = cat(2,featuresValidation,gather(extractdata(features)));
            end
            featuresValidation = featuresValidation.';
            
            disp("Defining neural network ...");
            % Define a deep learning network that classifies the feature vectors.
            
            numFeatures = mdl.Parameters.Hyperparameters.HiddenSize;
            numClasses = numel(categories(YTrain));
            layers = [
                featureInputLayer(numFeatures)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
            
            % Specify the training options using the |trainingOptions| function.
            % * Train with a mini-batch size of 64.
            % * Shuffle the data every epoch.
            % * Validate the network using the validation data.
            % * Display the training progress in a plot and suppress the verbose
            %   output.
            opts = trainingOptions('adam',...
                "MiniBatchSize",64,...
                "ValidationData",{featuresValidation,YVal}, ...
                "Shuffle","every-epoch", ...
                "Plots","training-progress", ...
                "Verbose",0);
            
            disp("Training network ...");
            % Train the network using the |trainNetwork| function.
            net = trainNetwork(featuresTrain,YTrain,layers,opts);
        end

        function [YPred,accuracy] = predict(obj,mdl,net,testTextArray,YTest)
            if nargin < 4
                YTest = [];
            else
                YTest = categorical(YTest);
            end
            if iscell(testTextArray) && isa(testTextArray{1},'double')
                disp("Input X is already tokenized. Skipping this step.");
                tokens = testTextArray;
            elseif isstring(testTextArray)
                % Tokenize the text data using the same steps as the training documents.
                tokens = encode(mdl.Tokenizer,testTextArray);
            else
                error('Input X must be either an array of strings or a cell array of doubles (=tokenized).');
            end
            
            % Pad the sequences of tokens to the same length using the |padsequences| 
            % function and pad using the tokenizer padding code.
            paddedX = padsequences(tokens,2,"PaddingValue",mdl.Tokenizer.PaddingCode);
            
            % Classify the new sequences using the trained model.
            featuresNew = bertEmbed(obj,paddedX,mdl.Parameters)';
            featuresNew = gather(extractdata(featuresNew));
            YPred = classify(net,featuresNew);

            if ~isempty(YTest)
                figure
                confusionchart(YTest,YPred);
                
                % Calculate the validation accuracy.
                accuracy = mean(YTest == YPred);
            else
                accuracy = [];
            end
        
        end


    end
    methods(Access=private)
        %%% Predictors Preprocessing Functions
        % The |preprocessPredictors| function truncates the mini-batches to have
        % the specified maximum sequence length, pads the sequences to have the
        % same length. Use this preprocessing function to preprocess the predictors
        % only.
        function X = preprocessPredictors(~,X,paddingValue,maxSeqLen)
            X = truncateSequences(X,maxSeqLen);
            X = padsequences(X,2,"PaddingValue",paddingValue);
        end
        
        %%% BERT Embedding Function
        % The |bertEmbed| function maps input data to embedding vectors and
        % optionally applies dropout using the "DropoutProbability" name-value
        % pair.
        function Y = bertEmbed(obj,X,parameters,args)
            arguments
                obj
                X
                parameters
                args.DropoutProbability = 0
            end
            
            dropoutProbabilitiy = args.DropoutProbability;
            
            Y = bert.model(X,parameters, ...
                "DropoutProb",dropoutProbabilitiy, ...
                "AttentionDropoutProb",dropoutProbabilitiy);
            
            % To return single feature vectors, return the first element.
            Y = Y(:,1,:);
            Y = squeeze(Y);
        end
    end
end