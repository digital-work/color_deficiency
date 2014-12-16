function [ success ] = c2g_alsam( args )
%C2G_ALSAM Summary of this function goes here
%   Detailed explanation goes here
    
    success = 1;
    img_in_path = args.path_in;
    img_out_path = args.path_out;
    
    disp(img_in_path);
    disp(img_out_path);
    
    img_in = imread(img_in_path);

    img_out = ali_grey_simpleonesided(img_in);
    
    imwrite(img_out,img_out_path)

end

