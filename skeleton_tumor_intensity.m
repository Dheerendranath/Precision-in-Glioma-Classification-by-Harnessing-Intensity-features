clear all
close all
% Set up the main directory and target subdirectories
main_directory = '/Users/dheerendranathbattalapalli/Desktop/TP/';
subdirs = 1:150;

% Loop over the subdirectories
for idx = 1:length(subdirs)
    subdir = subdirs(idx);
    folder_path = fullfile(main_directory, num2str(subdir));
    
    % Find the relevant NIfTI files in the current folder
    t1ce_file = dir(fullfile(folder_path, '*T1ce.nii'));
    label_file = dir(fullfile(folder_path, '*All-label.nii'));
    
    % If the NIfTI files are missing, continue to the next folder
    if isempty(t1ce_file) || isempty(label_file)
        fprintf('Missing T1ce.nii or All-label.nii in folder: %s\n', folder_path);
        continue;
    end
    
    t1ce_path = fullfile(folder_path, t1ce_file(1).name);
    label_path = fullfile(folder_path, label_file(1).name);
    
    % Load the NIfTI files
    t1ce_data = niftiread(t1ce_path);
    label_data = niftiread(label_path);
    
    % Convert the label data to a skeletal image
    skeleton_data = bwskel(logical(label_data));
    
    % Overlay the skeletal image on the T1ce image
    if isa(t1ce_data, 'double')

        overlay_image = t1ce_data .* double(skeleton_data);
    else
        overlay_image = int16(t1ce_data) .* int16(skeleton_data);
    end

    
    % Extract the first-order statistical values from the ROI
    roi_pixels = overlay_image(skeleton_data > 0);
    %mean_val = mean(roi_pixels);
    roi_pixels_double = double(roi_pixels);
    
    %variance_val = var(roi_pixels_double);

    skewness_val = skewness(double(roi_pixels));
    kurtosis_val = kurtosis(double(roi_pixels));
    entropy_val = entropy(double(roi_pixels));
    min_val = min(roi_pixels);
    max_val = max(roi_pixels);
    range_val = range(roi_pixels);
    mad_val = mad(roi_pixels, 1);
    rms_val = rms(roi_pixels);
    total_energy_val = sum(roi_pixels.^2);
    uniformity_val = sum(histcounts(roi_pixels).^2)/numel(roi_pixels);
    variance_val = var(double(roi_pixels));
    std_val = std(double(roi_pixels));
    mean_val = mean(double(roi_pixels));
    cv_val = std_val/mean_val; % Coefficient of variation
    % Extract more first-order statistical values from the ROI
    mode_val = mode(roi_pixels_double);
    median_val = median(roi_pixels_double);
    iqr_val = iqr(roi_pixels_double);
    perc10_val = prctile(roi_pixels_double, 10);
    perc90_val = prctile(roi_pixels_double, 90);
    energy_val = sum(roi_pixels_double.^2);
    robust_mad_val = mad(roi_pixels_double(roi_pixels_double >= perc10_val & roi_pixels_double <= perc90_val), 1);

    
    % Save the extracted values in a text file for each folder
    outfile = fullfile(folder_path, 'first_skeletal_t1ce_order_statistics.txt');
    fid = fopen(outfile, 'w');
    fprintf(fid, 'Mean: %f\n', mean_val);
    fprintf(fid, 'Variance: %f\n', variance_val);
    fprintf(fid, 'Skewness: %f\n', skewness_val);
    fprintf(fid, 'Kurtosis: %f\n', kurtosis_val);
    fprintf(fid, 'Entropy: %f\n', entropy_val);
    fprintf(fid, 'Minimum: %d\n', min_val);
    fprintf(fid, 'Maximum: %d\n', max_val);
    fprintf(fid, 'Range: %d\n', range_val);
    fprintf(fid, 'Mean Absolute Deviation: %f\n', mad_val);
    fprintf(fid, 'Root Mean Square: %f\n', rms_val);
    fprintf(fid, 'Total Energy: %f\n', total_energy_val);
    fprintf(fid, 'Uniformity: %f\n', uniformity_val);
    fprintf(fid, 'Coefficient of Variation: %f\n', cv_val);
    fprintf(fid, 'Mode: %f\n', mode_val);
    fprintf(fid, 'Median: %f\n', median_val);
    fprintf(fid, 'Interquartile Range: %f\n', iqr_val);
    fprintf(fid, '10th Percentile: %f\n', perc10_val);
    fprintf(fid, '90th Percentile: %f\n', perc90_val);
    fprintf(fid, 'Energy: %f\n', energy_val);
    fprintf(fid, 'Robust Mean Absolute Deviation: %f\n', robust_mad_val);
    fclose(fid);
end

fprintf('Feature extraction completed!\n');
