classdef SentimentAnalysis
    %SENTIMENTANALYSIS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Property1
    end
    
    methods
        function obj = SentimentAnalysis(inputArg1,inputArg2)
            %SENTIMENTANALYSIS Construct an instance of this class
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

    end
end

