function g = ali_grey_simpleonesided(im)
im=im2double(im);
[M,N,n3]=size(im);
[imx,imy]=gradient(im);
close all;

imx = reshape(imx,M*N,3);
imy = reshape(imy,M*N,3);
[maxx,indxx] = max(abs(imx),[],2);
[maxy,indxy] = max(abs(imy),[],2);
% now max-gradients:
themax_x = zeros(M*N,1); themax_y = zeros(M*N,1);
for i=1:(M*N)
  themax_x(i) = imx(i,indxx(i));
  themax_y(i) = imy(i,indxy(i));
end
themax_x = reshape(themax_x,M,N);
themax_y = reshape(themax_y,M,N);
%g = forwardandback(themax_x,themax_y);
g = frankotchellappa(themax_x,themax_y);
g =g-min(min(g));
g=g./max(max(g));
g=imadjust(g);
%figure
%imshow(g);
