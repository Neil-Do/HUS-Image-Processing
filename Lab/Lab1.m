pkg load image

# Ham Cau 3
function retval = ChromaKeying(I)
  data = dlmread("data.txt");
  #id_ = find(data(:, 4));

  #K size (791 1064 3);
  K = zeros(size(I));

  # [row, col, channel] = [791 1064 3];
  [row, col, channel] = size(I);

  for j = 1:col
    count = 1;
    for i = ((j - 1) * row + 1) : (j * row)
      if data(i, 4) == 0
        K(count, j, :) = data(i, 1:3);
        count = count + 1;
      endif
    endfor
  endfor

  K1 = uint8(K);
  retval = K1;
endfunction
# End Ham Cau 3



# Ham main

# Cau 1
I = imread("lion.jpg");
J = imread("lionHSV.jpg");

%{
subplot(1, 2, 1);
imshow(I);
subplot(1, 2, 2);
imshow(J);
%}
# Het Cau 1

# Cau 2
%{
Jh = J(:, :, 1);
Js = J(:, :, 2);
Jv = J(:, :, 3);

Jh_double = im2double(Jh)
B = (Jh_double > 0.22 & Jh_double < 0.45)
imshow(B)
%}
# Het Cau 2

# Cau 3
K = ChromaKeying(I);
imshow(K);

%{
subplot(1, 2, 1);
imshow(I);
subplot(1, 2, 2);
imshow(K);
%}
# Het Cau 3
