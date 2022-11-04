classdef Stats
    %STATISTICS Summary of this class goes here
    %   Detailed explanation goes here
    
    methods(Static)       
        function plotLabelDistribution(data)
            label = categorical(data.Label);
            figure;
            histogram(label);
            xlabel("Class");
            ylabel("Frequency");
            title("Class Distribution");
        end

        function plotDocLength(documents)
            documentLengths = doclength(documents);
            figure
            histogram(documentLengths)
            title("Document Lengths")
            xlabel("Length")
            ylabel("Number of Documents")
        end

    end
end

