classdef FeatureExtraction
    %FEATUREEXTRACTION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Property1
    end
    
    methods
        function obj = FeatureExtraction(inputArg1,inputArg2)
            %FEATUREEXTRACTION Construct an instance of this class
            %   Detailed explanation goes here
            obj.Property1 = inputArg1 + inputArg2;
        end

        function documents = preprocessText(textData)           
            % Tokenize.
            documents = tokenizedDocument(textData);
            
            % Remove tokens containing digits.
            pat = textBoundary + wildcardPattern + digitsPattern + wildcardPattern + textBoundary;
            documents = replace(documents,pat,"");
            
            % Convert to lowercase.
            documents = lower(documents);
            
            % Remove short words.
            documents = removeShortWords(documents,2);
            
            % Remove stop words.
            documents = removeStopWords(documents);
        end
        
        function [bag,rmIdx] = getBagOfWords(~,documents)
            bag = bagOfWords(documents);
            % Remove words from the bag-of-words model that do not appear more than two times in total. Remove any documents containing no words from the bag-of-words model, and remove the corresponding entries in labels.
            bag = removeInfrequentWords(bag,2);
            [bag,rmIdx] = removeEmptyDocuments(bag);
        end

        function outputArg = getTFIDF(obj)
            %METHOD1 Summary of this method goes here
        end

        function enc = wordToVec(~,documents)
            %METHOD1 Summary of this method goes here
            enc = wordEncoding(documents);
        end


    end
end

