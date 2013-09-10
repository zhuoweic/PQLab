function codes = codingPipeline(imdb, featuresPath, resultPath, varargin)

% --------------------------------------------------------------------
%                                                       Initialization
% --------------------------------------------------------------------
%% path for storing all the intermediate stuff
%% resultPath

%% low level extraction functions
opts.readImageFn = @readImage;

%% novel parameters for product quantization
opts.products = 1;
opts.transform = 'none';

%% codebooks formation parameters
opts.encoderParams = {'type', 'bovw'} ;
opts.numWords = [] ;
opts.numSamplesPerWord = [] ;
opts.type = 'bovw' ;
opts.layouts = {'1x1'} ;
opts.geometricExtension = 'none' ;

%% PCA dimension reduction & components normalization
opts.numPcaDimensions = +inf ;
opts.whitening = false ;
opts.whiteningRegul = 0 ;
opts.renormalize = false ;

%% light test option
opts.lite = false;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                        Features Extraction (for codebooks formation)
% --------------------------------------------------------------------

%% Initialization %%

%% number of training images
numTrain = 5000 ;
%% light option
if opts.lite
	numTrain = 10 ; 
end
%% training indexes
train = vl_colsubset(find(imdb.images.set <= 2), numTrain, 'uniform') ;
%% training images
imageList = cellfun(@(x) fullfile(imdb.imageDir, x), imdb.images.name(train), 'uniform', 0);

%% determine the total amount of features to be
%% extracted from all the images
if isempty(opts.numSamplesPerWord)
    switch opts.type
		case {'bovw'}
			opts.numSamplesPerWord = 200 ;
		case {'vlad', 'vlad_aug', 'fv'}
			opts.numSamplesPerWord = 1000 ;
		otherwise
			assert(false) ;
    end
    if opts.lite
		opts.numSamplesPerWord = 10 ;
    end
end

%% number of descriptors to be extracted from one image
numDescrsPerImage = ceil(opts.numWords * opts.numSamplesPerWord / numTrain) ;

if ~exist(fullfile(featuresPath, sprintf('trainingCodes#%d.mat', numDescrsPerImage)), 'file')  
	descrs = {};
	frames = {};
	parfor indexTrain = 1:numTrain
	% for indexTrain = 1:numTrain
		fprintf('%s: reading: %s\n', mfilename, imageList{indexTrain}) ;
		im = opts.readImageFn(imageList{indexTrain}) ;
		w = size(im,2) ;
		h = size(im,1) ;
		
		%% extract image features and store them in local disk
		[~, name, ~] = fileparts(imageList{indexTrain});
		imagePath = fullfile(featuresPath, strcat(name, '.mat'));
		if ~exist(imagePath, 'file')
			error('image feature: %s does not exist', imagePath);
		else 
		features = load(imagePath);
		end
		
		%% randomly select features from features pool of training data
		randn('state',0) ;
		rand('state',0) ;
		sel = vl_colsubset(1:size(features.descr,2), single(numDescrsPerImage)) ;
		descrs{indexTrain} = features.descr(:,sel) ;
		frames{indexTrain} = features.frame(:,sel) ;
		frames{indexTrain} = bsxfun(@times, bsxfun(@minus, frames{indexTrain}(1:2,:), [w;h]/2), 1./[w;h]) ;
	end
	descrs = cat(2, descrs{:}) ;
	frames = cat(2, frames{:}) ;
	disp('***** saving training codes *****');
	save(fullfile(featuresPath, sprintf('trainingCodes#%d.mat', numDescrsPerImage)), 'descrs', 'frames');
else
	disp('***** loading training codes *****');
	load(fullfile(featuresPath, sprintf('trainingCodes#%d.mat', numDescrsPerImage)), 'descrs', 'frames');
end

% --------------------------------------------------------------------
%                                                   Data Preprocessing
% --------------------------------------------------------------------

%% Initialization %%
opts.projection = 1 ;
opts.projectionCenter = zeros(size(descrs, 1), 1) ;

if ~exist(fullfile(resultPath, 'transform.mat'), 'file')  
  
	if strcmp(opts.transform, 'ica') == 0
		%%%%%%%%%%%%%%%%%%%%%%%%
		% learn PCA projection %
		%%%%%%%%%%%%%%%%%%%%%%%%
		if opts.numPcaDimensions < inf || opts.whitening
			fprintf('%s: learning PCA rotation/projection\n', mfilename) ;
			opts.projectionCenter = mean(descrs,2) ;
			%% subtract mean
			x = bsxfun(@minus, descrs, opts.projectionCenter) ;
			%% covariance matrix
			X = x*x' / size(x,2) ;
			%% obtain eigenvalues
			[V,D] = eig(X) ;
			d = diag(D) ;
			%% sort them (largest first)
			[d,perm] = sort(d,'descend') ;
			d = d + opts.whiteningRegul * max(d) ;
			m = min(opts.numPcaDimensions, size(descrs,1)) ;
			V = V(:,perm) ;
			if opts.whitening
				opts.projection = diag(1./sqrt(d(1:m))) * V(:,1:m)' ;
			else
				opts.projection = V(:,1:m)' ;
			end
			clear X V D d ;
		end

		%% apply PCA
		descrs = opts.projection * bsxfun(@minus, descrs, opts.projectionCenter) ;
		%% renormalise using L2 form
		if opts.renormalize
			normalizeCoef = 1./max(1e-12, sqrt(sum(descrs.^2))) ;
			descrs = bsxfun(@times, descrs,  normalizeCoef);
		end

		%% obtaining feature dimension such as SIFT, MNIST, etc.
		featureDim = size(descrs, 1);
		%% by default a identity matrix
		transform = eye(featureDim);

		%% apply linear transformation
		switch opts.transform
			case 'randomPerm', rand('state', 0); transform = transform(randperm(featureDim), :);
			case 'randomRotate', randn('state', 0); transform = randn(featureDim, featureDim); [U, ~, ~] = svd(transform); transform = U';
			case 'optimized-p', transform = eigenvalue_allocation(descrs', opts.products)';
		end
		
		%% apply linear transformation
		descrs = transform * descrs;
			
		projection = opts.projection ;
		projectionCenter = opts.projectionCenter;
		save(fullfile(resultPath, 'transform.mat'), 'projection', 'projectionCenter', 'transform');
		
	else
		%%%%%%%%%%%%%%%%%%%%%%%%
		% learn ICA components %
		%%%%%%%%%%%%%%%%%%%%%%%%
		opts.projectionCenter = mean(descrs,2) ;
		%% subtract mean
		descrs = bsxfun(@minus, descrs, opts.projectionCenter) ;
		%% apply ica
		[descrs, ~, transform] = fastica(descrs, 'numOfIC', opts.numPcaDimensions, 'approach', 'symm');
		%% descrs and transform matrix are now a double, we need to convert it into a single one
		descrs = single(descrs);
		transform = single(transform);
		%% new feature dimension
		featureDim = size(descrs, 1);
		
		projection = opts.projection ;
		projectionCenter = opts.projectionCenter;
		save(fullfile(resultPath, 'transform.mat'), 'transform', 'projectionCenter', 'projection');
	end
	
else
	%% if exist transform, do nothing
	disp('***** loading transform matrix *****');
	load(fullfile(resultPath, 'transform.mat'));
	opts.projection = projection;
	opts.projectionCenter = projectionCenter;
    descrs = transform * opts.projection * bsxfun(@minus, descrs, opts.projectionCenter) ;
end

%% class of descriptors and transform must be single 
%% convert it if necessary
if strcmp(class(descrs), 'single') == 0
	error('descriptors type must be single.');
end
% --------------------------------------------------------------------
%                                                Subspace Partitioning
% --------------------------------------------------------------------

%% partition descriptors into a cell
%% each component represents a partitioned part
%% the resulting partition is an index cell
%% can incorporate different coding method in different subspaces
partition = {};
for subspace = 1 : opts.products
    subdim = size(descrs, 1) / opts.products;
    partition{subspace} = [1 : subdim] + (subspace - 1) * subdim ;
end
numSubspaces = size(partition, 2);

% --------------------------------------------------------------------
%                                          Encoders/Codebooks Training
% --------------------------------------------------------------------

%% recording training time
tic;
if ~exist(fullfile(resultPath, 'encoder.mat'), 'file')  
	encoder = {};
	parfor subspace = 1 : numSubspaces
		encoder{subspace} = trainEncoder(descrs(partition{subspace}, :), frames, ...
							 opts.encoderParams{:}, ...
							 'lite', opts.lite) ;
	end
	save(fullfile(resultPath, 'encoder.mat'), 'encoder') ;
else 
	disp('***** loading encoder *****') ;
	load(fullfile(resultPath, 'encoder.mat')) ;
end

disp('***** training complete *****');
toc;
% --------------------------------------------------------------------
%                                 Features Extraction (for all images)
% --------------------------------------------------------------------

%% Initialization %%
%% all images list
imageList = cellfun(@(x) fullfile(imdb.imageDir, x), imdb.images.name, 'uniform', 0);
%% number of all the images
numImages = numel(imageList) ;
%% recoding coding time
tic;
%% encoded codes
codes = {} ;
%% progress reminder
parfor_progress(numImages, resultPath) ;

parfor indexAll = 1:numImages
% for indexAll = 1:numImages
	%% reading images
	% fprintf('%s: reading: %s\n', mfilename, imageList{indexAll}) ;
	im = opts.readImageFn(imageList{indexAll}) ;
	imageSize = size(im) ;
	w = size(im,2) ;
	h = size(im,1) ;
	
	%% extract image features and store them in local disk
	[~, name, ~] = fileparts(imageList{indexAll});
	imagePath = fullfile(featuresPath, strcat(name, '.mat'));
	if ~exist(imagePath, 'file')
		error('image feature: %s does not exist', imagePath);
	else 
		features = load(imagePath);
	end
	
	%% apply dimension reduction and regularization
	descr = features.descr;
	frame = features.frame;
	descr = transform * opts.projection * bsxfun(@minus, descr, opts.projectionCenter) ;
	
	%% separately encoding for different subspaces 
	code = cell(1, numSubspaces);
	for subspace = 1 : numSubspaces
		code{subspace} = encodeImage(descr(partition{subspace}, :), frame, imageSize, encoder{subspace});
	end
	
	%% connect codes 
	codes{indexAll} = cat(1, code{:});
	%% show progress
	parfor_progress(-1, resultPath);
end

%% close progress reminder
parfor_progress(0, resultPath);

codes = cat(2, codes{:});
%% store intermediate results
disp('***** saving all codes *****');
save(fullfile(resultPath, 'codes.mat'), 'codes', '-v7.3') ;

disp('***** images encode complete *****');
toc;