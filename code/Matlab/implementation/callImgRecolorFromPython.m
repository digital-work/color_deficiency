function a = callImgRecolorFromPython(args)
    a = [];
    path = args.path_tmp;
    coldef_type = args.coldef_type;
    if isfield(args,'from_python')
        from_python = true;
    else
        from_python = false;
    end
    
    if coldef_type == 'p'
        coldef_type = 'Protanopia';
    elseif coldef_type == 'd'
        coldef_type = 'Deuteranope';
    elseif coldef_type == 't'
        coldef_type = 'Tritanopia';
    end
    img_in = imread(path);
    disp(coldef_type)
    disp(path)
    disp(exist(path, 'file'))
    disp('Starting daltonization.')
    tic;
    [imgRecolorRGB, imgRecolorSimRGB, imgSimRGB] = imgRecolor(im2double(img_in), coldef_type);
    t = toc;
    disp(['Image recoloring done in ', num2str(t), ' seconds']);
    
    disp(from_python)
    if ~from_python
        %% Show images 
        figure(1)
        subplot(2,2,1), imshow(img_in);
        title('Original image');
        subplot(2,2,2), imshow(imgSimRGB);
        title(['Simulation of the origianl image: (', coldef_type,')']);
        subplot(2,2,3), imshow(imgRecolorRGB);
        title('Recolored image');
        subplot(2,2,4), imshow(imgRecolorSimRGB);
        title(['Simulation of the recolored image: (', coldef_type,')']);
    end
    
    imwrite(imgRecolorRGB,path,'png');
    %disp('hiersimmer3');
end