function fv_ex()
% EXPERIMENTS   Run image classification experiments
%    The experiments download a number of benchmark datasets in the
%    'data/' subfolder. Make sure that there are several GBs of
%    space available.
%
%    By default, experiments run with a lite option turned on. This
%    quickly runs all of them on tiny subsets of the actual data.
%    This is used only for testing; to run the actual experiments,
%    set the lite variable to false.
%
%    Running all the experiments is a slow process. Using parallel
%    MATLAB and several cores/machines is suggested.
%	 The procedure is set as a function in case to affect the main environment.

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

lite = true ;
clear ex ;

% initialization %%
datasetDir = 'd:\workworkwork\vlfeat-0.9.17\apps\recognition\';
experimentDir = 'd:\workworkwork\';
% datasetDir = '/nfs/home/zhuowei/datasets' ;
% experimentDir = '/nfs/home/zhuowei/experiments' ;

%%%%%%%%%%%%%%%%%%%%%%%%
% fisher vector coding %
%%%%%%%%%%%%%%%%%%%%%%%%

%% voc2007
ex.prefix = 'fv' ;												% experiment type
ex.datasets = {'voc07'} ;										% images dataset
ex.seed = 1 ;													% random seed for reproducing work
ex.kernelOpts = {'kernel', 'linear'} ;							% kernel options
ex.svmOpts = {'C', 1} ;											% svm options
ex.productOpts = {'products', 1} ;								% number of subspaces
ex.encoderOpts = {...											% encoder options
  'type', 'fv', ...
  'numWords', 256, ...
  'layouts', {'1x1'}, ...
  'geometricExtension', 'none'};
ex.transformOpts = {...											% projection transform options
  'numPcaDimensions', 80, ...
  'transform', 'randomRotate'};
ex.extractorFn = @(x) getDenseSIFT(x, ...						% dense sift settings different 
                                   'step', 4, ...				% for caltech101 and others
                                   'scales', 2.^(1:-.5:-3));	% all but caltech101 doubled the resolution
%***** augment data? flip the images? multi-scale sift? *****%


%%%%%%%%%%%%%%%%%%%%%%%%
% classification start %
%%%%%%%%%%%%%%%%%%%%%%%%
dataset = ex.datasets{1} ;

if ~isfield(ex, 'svmOpts') || ~iscell(ex.svmOpts)
  ex.svmOpts = {} ;
end

%% tiny dataset tag
if lite
	tag = sprintf('%d#%d-%s', ex.productOpts{2} , ex.encoderOpts{4}, 'lite'); 
else
	tag = sprintf('%d#%d', ex.productOpts{2} , ex.encoderOpts{4}); 
end

traintest(...
	ex.kernelOpts{:}, ...
	ex.svmOpts{:}, ...
	ex.productOpts{:}, ...	
	'prefix', [dataset '-' ex.prefix '-' tag], ...
	'seed', ex.seed, ...
	'dataset', char(dataset), ...
	'experimentDir', experimentDir, ...
	'datasetDir', datasetDir, ...
	'lite', lite, ...
	'extractorFn', ex.extractorFn, ...
	'encoderParams', ex.encoderOpts, ...
	'transformParams', ex.transformOpts) ;
end
