classdef Statistics
    %STATISTICS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = Statistics(data)
            %STATISTICS Construct an instance of this class
            %   Detailed explanation goes here
        end
        
        function plotLabelDistribution(obj,data)
            label = categorical(data.Label);
            figure;
            histogram(label);
            xlabel("Class");
            ylabel("Frequency");
            title("Class Distribution");
        end

        function plotDocLength(obj,documents)
            documentLengths = doclength(documents);
            figure
            histogram(documentLengths)
            title("Document Lengths")
            xlabel("Length")
            ylabel("Number of Documents")
        end

    end
end

