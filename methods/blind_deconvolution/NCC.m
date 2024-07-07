function ncc = NCC(x, y)
    x_mean = mean(x(:));
    y_mean = mean(y(:));
    x_std  = std(x(:));
    y_std  = std(y(:));
    ncc = (x-x_mean).*(y-y_mean)./(x_std*y_std);
    ncc = mean(ncc(:));
end