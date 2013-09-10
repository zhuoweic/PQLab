function code = encodeImage(descr, frame, imageSize, encoder, varargin)
% ENCODEIMAGE   Apply an encoder to an image
%   DESCRS = ENCODEIMAGE(ENCODER, IM) applies the ENCODER
%   to image IM, returning a corresponding code vector PSI.
%
%   IM can be an image, the path to an image, or a cell array of
%   the same, to operate on multiple images.
%
%   ENCODEIMAGE(ENCODER, IM, CACHE) utilizes the specified CACHE
%   directory to store encodings for the given images. The cache
%   is used only if the images are specified as file names.
%
%   See also: TRAINENCODER().

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%% initialization %%
features.descr = descr;
features.frame = frame;
code = {} ;

for i = 1:size(encoder.subdivisions,2)
	minx = encoder.subdivisions(1,i) * imageSize(2) ;
	miny = encoder.subdivisions(2,i) * imageSize(1) ;
	maxx = encoder.subdivisions(3,i) * imageSize(2) ;
	maxy = encoder.subdivisions(4,i) * imageSize(1) ;

	ok = ...
		minx <= features.frame(1,:) & features.frame(1,:) < maxx  & ...
		miny <= features.frame(2,:) & features.frame(2,:) < maxy ;

	descrs =  features.descr(:,ok);

	w = imageSize(2) ;
	h = imageSize(1);
	frames = features.frame(1:2,:) ;
	frames = bsxfun(@times, bsxfun(@minus, frames, [w;h]/2), 1./[w;h]) ;

	descrs = extendDescriptorsWithGeometry(encoder.geometricExtension, frames, descrs) ;

	switch encoder.type
		case 'bovw'
			[words,distances] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
											descrs, ...
											'MaxComparisons', 100) ;
			z = vl_binsum(zeros(encoder.numWords,1), 1, double(words)) ;
		case 'fv'
			z = vl_fisher(descrs, ...
						encoder.means, ...
						encoder.covariances, ...
						encoder.priors, ...
						'Improved') ;
		case 'vlad'
			[words,distances] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
											descrs, ...
											'MaxComparisons', 15) ;
			assign = zeros(encoder.numWords, numel(words), 'single') ;
			assign(sub2ind(size(assign), double(words), 1:numel(words))) = 1 ;
			z = vl_vlad(descrs, ...
						encoder.words, ...
						assign, ...
						'SquareRoot', ...
						'NormalizeComponents') ;
		case 'vlad_aug'
			[words, distances] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
											descrs, ...
											'MaxComparisons', 15) ;
			%% obtain real centers
			centers = zeros(size(descrs, 1), encoder.numWords);
			for sample = 1 : numel(words)
				centers(:, words(sample)) = centers(:, words(sample)) + descrs(:, sample) ;
			end
			%% obtain centers
			for component = 1 : encoder.numWords
				centers(:, component) = centers(:, component) / max(sum(words == component), 1) ;
			end
			
			%% obtain center-mean deviation
			z = zeros(size(descrs, 1), encoder.numWords);		
			for sample = 1 : numel(words)
				z(:, words(sample)) = z(:, words(sample)) + descrs(:, sample) - encoder.words(:, words(sample)) ;

			end
			%% power normalization (alpha = 2)
			z = sign(z) .* sqrt(abs(z));
			%%  L2 component-wise normalization
			for component = 1 : encoder.numWords
				z(:, component) = z(:, component) / max(norm(z(:, component)), 1e-12);
			end
			
			%% obtain centers deviation in various norms
			aug_z = zeros(size(descrs, 1), encoder.numWords);		
			for sample = 1 : numel(words)
				aug_z(:, words(sample)) = aug_z(:, words(sample)) + ...
									abs(descrs(:, sample) - centers(:, words(sample))) - ...
									abs(descrs(:, sample) - encoder.words(:, words(sample))) ;
			end
			%% power normalization (alpha = 2)
			aug_z = sign(aug_z) .* sqrt(abs(aug_z));
			%%  component-wise normalization
			for component = 1 : encoder.numWords
				aug_z(:, component) = aug_z(:, component) / max(norm(aug_z(:, component)), 1e-12);
			end
			z = cat(2, z, aug_z) ;

			%% double type descriptors cannot further improve performance
			z = single(z);
	end
	%% overall normalization on the pooled code
	z = z / max(sqrt(sum(z.^2)), 1e-12) ;
	code{i} = z(:) ;
end

code = cat(1, code{:}) ;