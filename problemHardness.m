risTable = readtable('./data/categorized-possible-hemorrhages-after-thrombectomy.csv');
% risTable = readtable('./data/all-post-thrombectomy-patients.csv')
question = risTable.Fragestallning;
freeText1 = risTable.Remisstext;
freeText2 = risTable.Prioanteckning;
freeText3 = risTable.Utlatandetext;
scanner = risTable.Modalitetsnamn;
label = risTable.Hemorrhage_;
idx = ~isnan(label);

freeTextTable = risTable(:,[4,5,6]);

% join all text here (columns must be cell arrays of characters)
colJoin =  rowfun(@(a,b,c) {strjoin([a,b,c],' ')}, freeTextTable, 'InputVariables', {'Remisstext','Prioanteckning','Utlatandetext'});
fullJoin = string(strjoin(colJoin{:,1},' '));

data = [colJoin,num2cell(label)];
data.Properties.VariableNames = ["X","Y"];
data = convertvars(data,{'X','Y'},'string');
dataLabeled = data(idx,:);

% use (1) "no hemorrhage" as "excluding label"
dataLabeled.Y(strcmp(dataLabeled.Y,'5')) = '1'; % summarize "no scan performed" as "excluding label"
dataLabeled.Y(strcmp(dataLabeled.Y,'3')) = '1'; % summarize "unclear" as "excluding label" 
dataLabeled.Y(strcmp(dataLabeled.Y,'4')) = '1'; % summarize "hemorrhage already on earlier scan" as "excluding label"

listAccW = [];
listAucW = [];
listSensW = [];
listSpecW = [];

listAccWO= [];
listAucWO= [];
listSensWO= [];
listSpecWO= [];

percList = [0.5,0.6,0.7,0.8,0.9,1];
percAcc = [];
percSens = [];
percSpec = [];
for j = 1:length(percList)
    for i = 1:10
        % Training
        classObj = Classification();
        % Generates a random split on every iteration
        
        percIter = percList(j);
        [trainArray,trainLabels,testArray,testLabels,~,~] = classObj.splitTrainValTest(dataLabeled,0.8 * percIter,0.2 * percIter);
        disp('Using ' + string(length(trainArray) + ' training samples ...'))
    
        trainDocs = classObj.preprocessText(trainArray,'swedish');
        [XTrain,bags] = classObj.extractFeatures(trainDocs);
        
        idxBest = classObj.selectFeatures(XTrain,trainLabels);
        XTrain = XTrain(:,idxBest);
        
        target.class = '2';
        target.metric = 'sens';
        target.value = 0.95;
        [model,target] = classObj.train(XTrain,trainLabels,target);
        
        % Test
        testDocs = classObj.preprocessText(testArray,'swedish');
        specs.bags = bags;
        specs.model = model;
        specs.idxBest = idxBest;
        specs.target = target;
        
        specs.target = [];
        specs.target.class = '2';
        [~,statsWithoutTarget] = classObj.predict(specs,testDocs,testLabels);
        listAccWO = [listAccWO,statsWithoutTarget.acc];
        listAucWO = [listAucWO,statsWithoutTarget.auc];
        listSensWO = [listSensWO,statsWithoutTarget.sens];
        listSpecWO = [listSpecWO,statsWithoutTarget.spec];
    end
    
    avgAccWO = mean(listAccWO);
    avgAucWO = mean(listAucWO);
    avgSensWO = mean(listSensWO);
    avgSpecWO = mean(listSpecWO);
    
    stdSpecWO = std(listAccWO);
    stdSpecWO = std(listAucWO);
    stdSpecWO = std(listSensWO);
    stdSpecWO = std(listSpecWO);
    
    percAcc = [percAcc,avgAccWO];
    percSens = [percSens,avgSensWO];
    percSpec = [percSpec,avgSpecWO];
end