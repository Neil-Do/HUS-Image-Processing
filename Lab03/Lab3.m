clear
pkg load image

I1 = imread("barCodesDetection.png");
IG = rgb2gray(I1);
imshow(IG);
J = (IG(:, :) <= 120);
imshow(J);

# algorithm TwoPass : https://en.wikipedia.org/wiki/Connected-component_labeling
%{
function retval = neighbors_labels(J, r, c)
  old_neighbors = [J(r-1,c-1), J(r-1,c), J(r-1,c+1), J(r, c-1)];
  old_labels = old_neighbors(old_neighbors > 0);
  if length(old_labels) == 0
    retval = [];
  else
    retval = unique(old_labels);
  endif
endfunction

function labels = ConnectedComponent(J)
  # linked matrix save equivalent labels
  linked = zeros(100, 1000);
  [rows, cols] = size(J);
  labels = zeros(rows, cols);
  nextLabel = 1;

  # first pass
  for r = 1:rows
    for c = 1:cols
      if J(r, c) == 1
        old_labels = neighbors_labels(J, r, c);
        if length(old_labels) == 0
          labels(r,c) = nextLabel;
          linked(nextLabel, 1) = nextLabel;
          # linked(:, 1000) = index of last label
          linked(:, 1000) = 1;
          nextLabel += 1;
        else
          min_label = min(old_labels)
          labels(r,c) = min_label;
          n = length(old_labels);
          # first label is min label
          for i = 2:n
            #add label in equivalent labels matrix
            index_of_last_label = linked(old_labels(i), 1000);
            linked(old_labels(i), index_of_last_label + 1) = min_label;
            linked(old_labels(i), 1000) += 1;
        endif
      endif
    endfor
  endfor

  # second pass
  for r = 1:rows
    for c = 1:cols
      if J(r, c) == 1
        currentLabel = labels(r, c);
        labels(r, c) = min(unique(linked(currentLabel, 1:999)));
    endfor
  endfor
endfunction
%}
