V = h5read('retina_output_gpot.h5');
aviobj = avifile('video.avi', 'compression', 'None');
for ii = 1:10000
    temp = V(:, ii);
    A = zeros(16*3,8*2); % 16 rows x 8 cols times....6?
    for jj = 1:(length(temp))
        if(temp(jj) > 0)
            A(jj) = 1;
        else
            A(jj) = 0;
        end
    end
    F = mat2gray(A);
    aviobj = addframe(aviobj, F);
end

aviobj = close(aviobj);
        
