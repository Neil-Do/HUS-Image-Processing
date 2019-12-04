clear
pkg load image

I = imread("lena512color.jpg");
#imshow(I);
[rowsI, colsI, channelI] = size(I);
J = zeros(size(I));
[rowsJ, colsJ, channelJ] = size(J);

ro0 = min(rowsJ, colsJ) / 2;
d0 = max(rowsI, colsI) / 2;
disp(ro0);
# tam hinh I
oI = round(colsJ / 2);

# tam hinh J
o_x = round(colsJ / 2);
o_y = round(rowsJ / 2);
for r = 1 - ro0 : rowsJ - ro0
  for c = 1 - ro0 : colsJ - ro0
    if c == 0 && r == 0
      J(r + ro0,c + ro0, :) = I(oI, oI, :);
    elseif r**2 + c**2  <= ro0**2
      ro = sqrt(r**2 + c**2);
      if c == 0 && r < 0
        theta = pi / 2;
      elseif c == 0 && r > 0
        theta = - pi / 2;
      else
        theta = atan(r/c);
      endif
      phi = asin(ro/ro0);
      d = (2/pi) * d0 * phi;
      r_in = round(d * sin(theta)) + oI;
      c_in = round(d * cos(theta)) + oI;
      r_in = max(1, r_in);
      r_in = min(rowsI, r_in);
      c_in = max(1, c_in);
      c_in = min(colsI, c_in);
      r_in = uint8(r_in);
      c_in = uint8(r_in);
      r_out = uint8(r + ro0);
      c_out = uint8(c + ro0);
      J(r_out,c_out, :) = I(r_in, c_in, :);
    endif
  endfor
endfor

imshow(J);
