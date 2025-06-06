img = imread("projectimages\[image.tif]]");
processSatelliteImage(img);

%% Main Processing Function
function enhanced_image = processSatelliteImage(img)
    figure('Name','Satellite Image Processing Comparison');

    % Show Original Color Image
    subplot(2,3,1), imshow(img), title('Original Image', 'FontWeight','bold');

    % Convert to grayscale
    if size(img,3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    img_gray = im2double(img_gray);

    % FFT
    F = fft2(img_gray);
    Fshift = fftshift(F);
    magnitude_spectrum = log(1 + abs(Fshift));
    subplot(2,3,2), imshow(magnitude_spectrum, []), title('FFT Spectrum', 'FontWeight','bold');

    [M, N] = size(img_gray);

    % Auto-detect peaks from FFT magnitude spectrum
    [u_k, v_k] = detectFFTpeaks(magnitude_spectrum, 2);  % get the 2 strongest peaks

    % Filtering parameters
    D0 = 10;
    n = 2;

    % Butterworth Notch Filtering
    H_butter = ones(M, N);
    for k = 1:length(u_k)
        H_butter = H_butter .* butterworthNotchReject(M, N, u_k(k), v_k(k), D0, n);
        H_butter = H_butter .* butterworthNotchReject(M, N, -u_k(k), -v_k(k), D0, n);
    end
    subplot(2,3,3), imshow(H_butter, []), title('Butterworth Filter Mask', 'FontWeight','bold');

    F_butter = Fshift .* H_butter;
    img_butter = real(ifft2(ifftshift(F_butter)));
    img_butter = mat2gray(img_butter);
    img_butter_color = applyGrayToColor(img, img_butter);
    subplot(2,3,4), imshow(img_butter_color), title('Butterworth Notch', 'FontWeight','bold');

    % Gaussian High-Pass Filtering
    sigma = 7;
    [X, Y] = meshgrid(-N/2:N/2-1, -M/2:M/2-1);
    H_gauss = 1 - exp(-(X.^2 + Y.^2) / (2 * sigma^2));


    F_gauss = Fshift .* H_gauss;
    img_gauss = real(ifft2(ifftshift(F_gauss)));
    img_gauss = mat2gray(img_gauss);
    img_gauss_color = applyGrayToColor(img, img_gauss);
    subplot(2,3,5), imshow(img_gauss_color), title('Gaussian High-Pass', 'FontWeight','bold');

    % Histogram Equalization
    img_hist_eq = histeq(img_gray);
    img_hist_eq_color = applyGrayToColor(img, img_hist_eq);
    subplot(2,3,6), imshow(img_hist_eq_color), title('Histogram Equalized', 'FontWeight','bold');

    enhanced_image = img_hist_eq_color;
end

%% Butterworth Notch Reject Filter
function H = butterworthNotchReject(M, N, u_k, v_k, D0, n)
    [U, V] = meshgrid(1:N, 1:M);
    U = U - N/2;
    V = V - M/2;
    Dk = sqrt((U - u_k).^2 + (V - v_k).^2);
    Dk_ = sqrt((U + u_k).^2 + (V + v_k).^2);
    Dk(Dk == 0) = eps; Dk_(Dk_ == 0) = eps;
    H = 1 ./ (1 + (D0^2 ./ (Dk .* Dk_)).^(2*n));
end

%% Reapply Grayscale Intensity to Color Image
function colored_result = applyGrayToColor(color_img, gray_result)
    gray_result = im2double(gray_result);
    color_img = im2double(color_img);
    hsv_img = rgb2hsv(color_img);
    hsv_img(:,:,3) = gray_result;
    colored_result = hsv2rgb(hsv_img);
end

%% Auto-Detect FFT Peaks Function
function [u_peaks, v_peaks] = detectFFTpeaks(mag_spec, num_peaks)
    mag_spec = mat2gray(mag_spec);
    mag_spec(size(mag_spec,1)/2-10:size(mag_spec,1)/2+10, size(mag_spec,2)/2-10:size(mag_spec,2)/2+10) = 0; % mask center

    % Find brightest peaks
    [~, sorted_idx] = sort(mag_spec(:), 'descend');
    [v_all, u_all] = ind2sub(size(mag_spec), sorted_idx);

    u_peaks = u_all(1:num_peaks) - size(mag_spec,2)/2;
    v_peaks = v_all(1:num_peaks) - size(mag_spec,1)/2;
end
