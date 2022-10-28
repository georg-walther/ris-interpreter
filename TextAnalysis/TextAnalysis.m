classdef TextAnalysis
    %TEXTANALYSIS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Property1
    end
    
    methods
        function obj = TextAnalysis(inputArg1,inputArg2)
            %TEXTANALYSIS Construct an instance of this class
            %   Detailed explanation goes here
            obj.Property1 = inputArg1 + inputArg2;
        end
        
        function documents = preprocess(textData)
            % Tokenize the text.
            documents = tokenizedDocument(textData);
            
            % Convert to lowercase.
            documents = lower(documents);
            
            % Erase punctuation.
            documents = erasePunctuation(documents);
        end



    end
end

