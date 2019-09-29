#I = imread("lion.jpg");
J = imread("lionHSV.jpg");

#subplot(1, 2, 1);
#imshow(I);
#subplot(1, 2, 2);
imshow(J);

Jh = J(:, :, 1);
Js = J(:, :, 2);
Jv = J(:, :, 3);

Jh_double = im2double(Jh);
#Jh_double = double(Jh) / double(max(Jh(:)));
Jh_bin = (0.22 < Jh_double & Jh_double < 0.45);
imshow(Jh_bin);
m = 1064
n = 791
data = dlmread("Data.txt");
K = zeros(m, n, 3);
for b = 1:3


Hoan thanh bai tap Lab1
Tang Gamma tren 
