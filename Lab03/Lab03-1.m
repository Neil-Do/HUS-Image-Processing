clear
pkg load image

# algorithm TwoPass : https://en.wikipedia.org/wiki/Connected-component_labeling


function retval = NeighborsLabels(J, r, c)
  old_neighbors = [J(r-1,c-1), J(r-1,c), J(r-1,c+1), J(r, c-1)];
  old_labels = old_neighbors(old_neighbors > 0);
  if length(old_labels) == 0
    retval = [];
  else
    retval = unique(old_labels);
  endif
endfunction


function P = AddPadding(J)
  # add padding
  [rows, cols] = size(J);
  P = zeros(rows + 2, cols + 2);
  P(2:rows+1, 2:cols+1) = J;
endfunction

function ConComI = PaintColor(B)
  # B = labels matrix
  # labels
  UniqueLabels = unique(B(B>0));
  max_label = max(UniqueLabels);
  color_lookup = zeros(max_label, 3);
  # #colors = 3 * n^2
  n = ceil(length(UniqueLabels)/3);
  base_color = floor(255/n);
  for i = 1:length(UniqueLabels)
    if i <= n
      color_lookup(UniqueLabels(i), :) = [base_color * i, 0, 0];
    elseif i <= 2 * n
      color_lookup(UniqueLabels(i), :) = [0, base_color * (i - n), 0];
    else
      color_lookup(UniqueLabels(i), :) = [0, 0, base_color * (i - 2*n)];
    endif
  endfor

  # pass matrix
  [rows, cols] = size(B);
  R = zeros(rows, cols, 3);
  for r = 1:rows
    for c = 1:cols
      if B(r, c) != 0
        R(r, c, :) = color_lookup(B(r, c), :);
      endif
    endfor
  endfor
  imshow(R);
  imwrite(R, "Results/PaintColor.png");
  ConComI = R;
endfunction


function labels = ConnectedComponent(J)
  # linked matrix save equivalent labels
  linked = zeros(100, 101);
  [rows, cols] = size(J);
  labels = zeros(rows, cols);
  labels = AddPadding(labels);
  nextLabel = 1;

  # first pass
  disp("Begin First Pass...")
  for r = 1:rows
    for c = 1:cols
      if J(r, c) == 1
        old_labels = NeighborsLabels(labels, r+1, c+1);
        if length(old_labels) == 0
          labels(r+1,c+1) = nextLabel;
          linked(nextLabel, 1) = nextLabel;
          # linked(:, 101) = index of last label
          linked(:, 101) = 1;
          nextLabel += 1;
        else
          min_label = min(old_labels);
          labels(r+1,c+1) = min_label;
          n = length(old_labels);
          # first label is min label
          for i = 2:n
            #add label in equivalent labels matrix
            index_of_last_label = linked(old_labels(i), 101);
            linked(old_labels(i), index_of_last_label + 1) = min_label;
            linked(old_labels(i), 101) += 1;
          endfor
        endif
      endif
    endfor
  endfor

  disp("End First Pass");
  # second pass
  disp("Begin Second Pass");
  count = 0;
  for r = 1:rows
    for c = 1:cols
      if J(r, c) == 1
        currentLabel = labels(r+1, c+1);
        equivalentLabels = linked(currentLabel, 1:100);
        equivalentLabels = unique(equivalentLabels(equivalentLabels > 0));
        labels(r+1, c+1) = min(equivalentLabels);
      endif
    endfor
  endfor
  disp("End Second Pass");
endfunction


# main script
%{
%}
I1 = imread("barCodesDetection.png");
IG = rgb2gray(I1);
#imshow(IG);
imwrite(IG, "Results/GrayImage.png");
J = (IG(:, :) <= 120);
#imshow(J);
imwrite(J, "Results/BinaryImage.png");

labels = ConnectedComponent(J);
csvwrite("Results/ConnectedComponentLabel.csv", labels);
ConComImage = PaintColor(labels);
# end main script

%{
  #test NeighborsLabels function
  A = [0, 0, 0; 0, 5, 6; 7, 8, 9];
  disp("A: ")
  disp(A);
  B = NeighborsLabels(A, 2, 2);
  if length(B) == 0
  disp("B null")
  else
  disp("B: ")
  disp(B);
  endif
%}

%{
  # test AddPadding function
  A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
  disp("A: ")
  disp(A);
  B = AddPadding(A);
  if length(B) == 0
  disp("B null")
  else
  disp("B: ")
  disp(B);
  endif
%}

%{
  # test ConnectedComponent function
  B = csvread("ConTestMatrix.csv");
  B = logical(B);
  labels = ConnectedComponent(B);
  csvwrite("Results/ConnectedComponentLabel.csv", labels);
%}

%{
  # test Paint Color Function
  B = csvread("PaintColorTest.csv");
  C = PaintColor(B);
%}
