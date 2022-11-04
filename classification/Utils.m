classdef Utils
    %UTILS Summary of this class goes here
    %   Detailed explanation goes here
    
    methods(Static)       
        function y = interpolate(x,x0,x1,y0,y1)
            y = (y0*(x1-x) + y1*(x-x0)) / (x1 - x0);
        end
    end
end

