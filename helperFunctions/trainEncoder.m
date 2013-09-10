function encoder = trainEncoder(descrs, frames, varargin)
% TRAINENCODER   Train image encoder: BoVW, VLAD, FV
%   ENCODER = TRAINENCOER(IMAGES) trains a BoVW encoder from the
%   specified list of images IMAGES.
%
%   TRAINENCODER(..., 'OPT', VAL, ...) accepts the following options:
%
%   Type:: 'bovw'
%     Bag of visual words ('bovw'), VLAD ('vlad') or Fisher Vector
%     ('fv').
%
%   Subdivisions:: []
%     A list of spatial subdivisions. Each column is a rectangle
%     [XMIN YMIN XMAX YMAX]. The spatial subdivisions are
%
%   Layouts:: {'1x1'}
%     A list of strings representing regular spatial subdivisions
%     in the format MxN, where M is the number of vertical
%     subdivisions and N the number of horizontal ones. For
%     example {'1x1', 2x2'} uses 5 partitions: the whole image and
%     four quadrants. The subdivisions are appended to the ones
%     specified by the SUBDIVISIONS option.
%
%   ReadImageFn:: @readImage
%     The function used to load an image.
%
%   ExtractorFn:: @getDenseSIFT
%     The function used to extract the feature frames and
%     descriptors from an image.
%
%	ENCODER parameters including
%		type:
%			bovw/vlad: words, kdtree
%			gmm: means, covariances, priors

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


%%%%%%%%%%%%%%%%%%
% initialization %
%%%%%%%%%%%%%%%%%%
%% encoder type
opts.type = 'bovw' ;
%% codebooks parameters
opts.numWords = [] ;
opts.numSamplesPerWord = [] ;
%% random seed for reproducing results
%% random stuff like k-means or Gmm (iteration method)
opts.seed = 1 ;
%% geometric information
opts.layouts = {'1x1'} ;
opts.geometricExtension = 'none' ;
 %***** subdivisions is used for layouts to generate spatial pyramids *****%
 %***** refer to encodeImage.m for further understanding of the skill *****%
opts.subdivisions = zeros(4,0) ;
%% feature extraction configuration
opts.readImageFn = @readImage ;
opts.extractorFn = @getDenseSIFT ;
%% tiny program option
opts.lite = false ;

opts = vl_argparse(opts, varargin) ;

%***** ? *****%
%% incorporating spatial information
for i = 1:numel(opts.layouts)
	t = sscanf(opts.layouts{i},'%dx%d') ;
	m = t(1) ;
	n = t(2) ;
	[x,y] = meshgrid(...
		linspace(0,1,n+1), ...
		linspace(0,1,m+1)) ;
	x1 = x(1:end-1,1:end-1) ;
	y1 = y(1:end-1,1:end-1) ;
	x2 = x(2:end,2:end) ;
	y2 = y(2:end,2:end) ;
	opts.subdivisions = cat(2, opts.subdivisions, ...
	[x1(:)' ;
	 y1(:)' ;
	 x2(:)' ;
	 y2(:)'] ) ;
end

disp(opts) ;

%% encoder initialization
encoder.type = opts.type ;
%% geometrical information
encoder.subdivisions = opts.subdivisions ;
encoder.geometricExtension = opts.geometricExtension ;
%% features extraction function
encoder.readImageFn = opts.readImageFn ;
encoder.extractorFn = opts.extractorFn ;
% codebook size
encoder.numWords = opts.numWords ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% adding geographic information %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% geometrically augment the features %%
descrs = extendDescriptorsWithGeometry(opts.geometricExtension, frames, descrs) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% iteration parameters can be modified %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% learn a VQ or GMM vocabulary %%
dimension = size(descrs,1) ;
numDescriptors = size(descrs,2) ;

%% [vl_twister] "control" randomness of the procedure
%% that is, it controls all states of VLFeat functions
disp('***** learning codebook *****');
switch encoder.type
	case {'bovw', 'vlad', 'vlad_aug'}
		vl_twister('state', opts.seed) ;
		encoder.words = vl_kmeans(descrs, opts.numWords, 'verbose', 'algorithm', 'elkan') ;
		encoder.kdtree = vl_kdtreebuild(encoder.words, 'numTrees', 2) ;

	case {'fv'} ;
		vl_twister('state', opts.seed) ;
		v = var(descrs')' ;
		[encoder.means, encoder.covariances, encoder.priors] = ...
			vl_gmm(descrs, opts.numWords, 'verbose', ...
				'Initialization', 'kmeans', ...
				'CovarianceBound', double(max(v)*0.0001), ...
				'NumRepetitions', 1) ;
    end
end
