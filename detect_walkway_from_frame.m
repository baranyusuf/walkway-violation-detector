function walkway_polygon = detect_walkway_from_frame(RGB)


    % Convert input frame to suitable format for processing
    %RGB = uint8(RGB_frame);  % Convert from MATLAB to RGB format

    BW = false(size(RGB,1),size(RGB,2));

    sam = segmentAnythingModel();  


    samImage = 255 * rescale(RGB);
    embeds = extractEmbeddings(sam, samImage);
    imsz = size(samImage, [1 2]);
    fgPoints = [];
    bgPoints = [];
    bbox = [];
    maskLogits = [];
    
    figure, imshow(RGB);
    title('Select Bounding Box');
    bbox = round(getrect);
    [BWout, ~, maskLogits] = segmentObjectsFromEmbeddings(sam, embeds, imsz, ...
        'ForegroundPoints', fgPoints, 'BackgroundPoints', bgPoints, 'BoundingBox', bbox, ...
        'MaskLogits', maskLogits);
    close;
    
    figure, imshow(RGB);
    hold on;
    overlay = cat(3, ones(size(BWout)), zeros(size(BWout)), zeros(size(BWout))); % Kırmızı maske
    h = imshow(overlay); % Maskeyi ekrana bas
    set(h, 'AlphaData', 0.5 * BWout);
    
    while true
        drawnow;
        title('Select 2 Foreground (left-click) or Background (right-click), press Enter to finish');
        while true
            [x, y, button] = ginput(1);
            if isempty(x)
                break;
            end
            
            if button == 1
                fgPoints(end+1, :) = [x y];
            elseif button == 3
                bgPoints(end+1, :) = [x y];
            elseif button == 97
                break;
            end
        end
        if isempty(x)
            close;
            break;
        end
        [BWout, ~, maskLogits] = segmentObjectsFromEmbeddings(sam, embeds, ...
            imsz, 'ForegroundPoints', fgPoints, 'BackgroundPoints', bgPoints, ...
            'BoundingBox', bbox, 'MaskLogits', maskLogits);
        set(h, 'AlphaData', 0.5 * BWout);
        drawnow; % Anında güncelle
    end
    
    BW = BW | BWout;
    maskedImage = RGB;
    maskedImage(repmat(~BW, [1 1 3])) = 0;
    
    % Extract walkway boundary
    boundaries = bwboundaries(BW);
    walkway_polygon = boundaries{1}; % Assuming the largest detected boundary is the walkway
end