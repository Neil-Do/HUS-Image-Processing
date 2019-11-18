%{
  Bai 1 cua bai thuc hanh Lab 2
%}

pkg load image


function retval = Convolution(I)
  [row, col, channel] = size(I)
  kernel = [0, -1, 0; -1, 5, -1; 0, -1, 0];
  I_uint16 = uint16(I);
  I5 = I_uint16 * 5;
  I_1_2 = I_uint16(1:row-1, :, :);
  I5(2:row, :, :) -= I_1_2;
  I_2_1 = I_uint16(:, 1:col-1, :);
  I5(:, 2:col, :) -= I_2_1;
  I_2_3 = I_uint16(:, 2:col, :);
  I5(:, 1:col-1, :) -= I_2_3;
  I_3_2 = I_uint16(2:row, :, :);
  I5(1:row-1, :, :) -= I_3_2;
  retval = uint8(I5);
endfunction

function retval = Pixelization(I)
  s = 8;
  [row, col, channel] = size(I);
  Jr = floor(row/s)
  Jc = floor(col/s);
  J = zeros([Jr, Jc, channel]);
  disp("here");
  for r = 1:Jr-1
    for c = 1:Jc-1
      T1 = I((r*s):(r+1)*s - 1, (c*s):(c+1)*s - 1, :);
      T = reshape(T1, s*s, 3);
      J(r, c, :) = mean(T);
    endfor
  endfor
  retval = uint8(J);
endfunction



# main

# Cau 1
I = imread("GreenVietnam.png");
S = Convolution(I);
S += I;
imwrite(S, "Result/GreenVietnamConvolution.png")
%{
subplot(1, 2, 1);
imshow(I);
subplot(1, 2, 2);
imshow(S);
%}
# Het Cau 1


# Cau 2
P = Pixelization(I);
imshow(P);
imwrite(P, "Result/GreenVietnamPixelization.png")
# Het Cau 2
