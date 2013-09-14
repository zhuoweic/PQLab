function mutualinformationMatrix = miMatrix(sample)

mutualinformationMatrix = zeros(size(sample, 1)) ;

for variableA = 1 : size(sample, 1)
	for variableB = 1 : variableA
		mutualinformationMatrix(variableA, variableB) = mi(sample(variableA, :)', sample(variableB, :)') ;
		mutualinformationMatrix(variableB, variableA) = mutualinformationMatrix(variableA, variableB) ;
		% fprintf('%dx%d\n', variableA, variableB) ;
	end
end