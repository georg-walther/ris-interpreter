function outText = translate(inText)
%TRANSLATE Summary of this function goes here
    space = {' '};
    actEnv = 'source activate languageEnv';
    status = system(actEnv);
    if status ~= 0
        status = system('conda create -y --name languageEnv python=3.7');
        cmd = strcat(actEnv,space,'&& conda install -y -c conda-forge transformers sentencepiece');
        status = system(cmd{1});
    end

    path = append('.',filesep,'translation',filesep,'translate.py');
    inText = strrep(inText,"`","'");
    inText = char(inText);
    seqLength = strlength(inText);
    maxLength = 512;
    sentenceEnds = strfind(inText,'.');
    endRefined = 0; % initialize
    outText = [];
    while endRefined < seqLength
        startIdx = endRefined + 1;
        endIdx = endRefined + maxLength;
        if ~isempty(sentenceEnds)
            [~,idx] = min(abs(sentenceEnds-endIdx));
            endRefined = sentenceEnds(idx);
            sentenceEnds(sentenceEnds <= endRefined) = []; % option unavailable in next iteration
        else
            endRefined = min([seqLength,endIdx]);
        end
        disp(string(startIdx) + ":" + string(endRefined))

        cmd = strcat(actEnv,space,'&& python',space,path,space,'"',inText(startIdx:endRefined),'"');
        [status,cmdOut] = system(cmd{1});
    
        partText = cmdOut;
        if ~strcmp(partText(1:4),'>>> ')
            error('Translation went wrong:' + cmdOut);
        else
            partText = partText(5:end);
        end
        outText = [outText,partText];
    end
end