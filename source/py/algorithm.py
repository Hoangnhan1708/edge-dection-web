import base64
import numpy as np
import cv2 

def gradient_library(image_data):
    try:
        _, image_data = image_data.split(';base64,')
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            return None
        
        # Áp dụng toán tử sobel với kích thước là 3
        gradient_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
        
        # Tính toán vecto cường độ gradient
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2).astype(int)
        magnitude = np.uint8(magnitude)
        
        # Tính toán BEI
        BEI = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)[1]
        
        # Trả về ảnh đã xử lý dưới dạng mảng NumPy
        return BEI
    except Exception as e:
        return str(e)

def gradient_custom(image_data):
    try:
        _, image_data = image_data.split(';base64,')
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            return None
        
        # Thiết lập kernel Sobel theo x và y
        kernel_x = (1 / 4) * np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = (1 / 4) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Thiết lập ngưỡng
        threshold = 50
        
        # Kích thước của ảnh
        m, n = gray_image.shape
        # Tạo ảnh cường độ của vector gradient với value = 0
        e = np.zeros_like(gray_image)
        # Tạo binary edge image với value = 0
        BEI = np.zeros_like(gray_image)
        # kích thước kernel
        kernel_size = len(kernel_x)
        # Kích thước padding
        pad_size = (kernel_size - 1) // 2
        # Padding với kích thước pad_size và value = 0
        pad_width = ((pad_size, pad_size), (pad_size, pad_size))
        padded_image = np.pad(gray_image, pad_width, mode='constant', constant_values=0)
        
        # Gán giá trị pixel vào e
        for x in range(pad_size, m + pad_size):
            for y in range(pad_size, n + pad_size):
                # Lấy các phần tử trong filter, phần tử đang xét nằm ở giữa
                neighbor = padded_image[x - pad_size:x + pad_size + 1, y - pad_size:y + pad_size + 1]
                # Tính ảnh cường độ
                f_x = np.sum(kernel_x * neighbor)
                f_y = np.sum(kernel_y * neighbor)
                e[x - pad_size][y - pad_size] = np.sqrt(f_x ** 2 + f_y ** 2)
        
        # Nếu cường độ vector gradient mà lớn hơn threshold thì đó là biên cạnh
        BEI[e > threshold] = 255
        return BEI
    except Exception as e:
        return str(e)

def laplace_library(image_data):
    try:
        _, image_data = image_data.split(';base64,')
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            return None
        laplacian_img = cv2.Laplacian(img_np, cv2.CV_8U, ksize=3)

        # Trả về ảnh đã xử lý dưới dạng mảng NumPy
        return laplacian_img
    except Exception as e:
        return str(e)

def laplace_custom(image_data):
    try:
        _, image_data = image_data.split(';base64,')
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            return None
        
        # Thiết lập kernel Laplace
        laplace_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        
        # Thiết lập ngưỡng T
        T = 600
        
        # Kích thước ảnh
        m, n = gray_image.shape
        # Khởi tạo ma trận lưu ảnh laplacian
        laplacian = np.zeros_like(gray_image)
        # Khởi tạo Binary Edge Image
        BEI = np.zeros_like(gray_image)
        # Kích thước padding
        pad_size = (len(laplace_filter) - 1) // 2
        # Padding với kích thước pad_size và value = 0
        pad_width = ((pad_size, pad_size), (pad_size, pad_size))
        padded_image = np.pad(gray_image, pad_width, mode='constant', constant_values=0)
        
        # Gán giá trị pixel vào laplacian
        for x in range(pad_size, m + pad_size):
            for y in range(pad_size, n + pad_size):
                # Lấy các phần tử trong filter, phần tử đang xét nằm ở giữa
                neighbor = padded_image[x - pad_size:x + pad_size + 1, y - pad_size:y + pad_size + 1]
                # Tính giá trị của ảnh laplacian
                laplacian[x - pad_size][y - pad_size] = np.sum(neighbor * laplace_filter)
        
        # Phát hiện biên cạnh
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                # Tìm điểm zero-crossing
                neighbors = [laplacian[i - 1, j], laplacian[i + 1, j], laplacian[i, j - 1], laplacian[i, j + 1]]
                if any(np.sign(laplacian[i, j]) != np.sign(neighbor) for neighbor in neighbors):
                    # Nếu điểm đó là zero crossing thì dùng cửa số 3x3 để tính local variance.
                    local_variance = np.var(gray_image[i - 1:i + 2, j - 1:j + 2])
                    # Nếu local variance > T thì đó là biên cạnh
                    if local_variance > T:
                        BEI[i, j] = 255
        
        return BEI
    except Exception as e:
        return str(e)

def canny_library(image_data):
    try:
        _, image_data = image_data.split(';base64,')
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            return None
        # Dùng gaussian để khử nhiễu
        blurred = cv2.GaussianBlur(img_np, (5, 5), 1)
        # Dùng canny
        edges = cv2.Canny(blurred, 20, 70)
        # Trả về ảnh đã xử lý dưới dạng mảng NumPy
        return edges
    except Exception as e:
        return str(e)
def laplacian_library(image_data):
    try:
        _, image_data = image_data.split(';base64,')
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            return None
        
        # Áp dụng LoG sử dụng OpenCV
        sigma = 1.5
        blurred_image = cv2.GaussianBlur(img_np, (0, 0), sigma)
        log_result = cv2.Laplacian(blurred_image, cv2.CV_64F)
        
        # TÌm ảnh BEI
        m, n = img_np.shape
        T = 500
        BEI = np.zeros_like(img_np)
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                # Tìm điểm zero-crossing
                neighbors = [log_result[i-1, j], log_result[i+1, j], log_result[i, j-1], log_result[i, j+1]]
                if any(np.sign(log_result[i, j]) != np.sign(neighbor) for neighbor in neighbors):
                    # Nếu điểm đó là zero crossing thì dùng cửa số 3x3 để tính local variance.
                    local_variance = np.var(img_np[i-1:i+2, j-1:j+2])
                    # Nếu local variance > T thì đó là biên cạnh
                    if local_variance > T:
                        BEI[i, j] = 255
        
        # Trả về ảnh đã xử lý dưới dạng mảng NumPy
        return BEI
    except Exception as e:
        return str(e)
    
def laplacian_of_gaussian(sigma, x, y):
    return -(1 / (np.pi * sigma**4)) * (1 - (x**2 + y**2) / (2 * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

def LoG_discrete(n, sigma, scale):
    LoG_filter = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            LoG_filter[i, j] = np.round(laplacian_of_gaussian(sigma, (i - (n - 1) / 2), (j - (n - 1) / 2)) * (-scale / laplacian_of_gaussian(sigma, 0, 0)))
    return LoG_filter

def laplacian_custom(image_data):
    try:
        _, image_data = image_data.split(';base64,')
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            return None
        
        # Thiết lập filter LoG
        LoG_filter = LoG_discrete(19, 1.5, 40)
        
        # Thiết lập ngưỡng T
        T = 500
        
        # Kích thước ảnh
        m, n = gray_image.shape
        # Khởi tạo LoG
        LoG = np.zeros_like(gray_image)
        # Khởi tạo Binary Edge Image
        BEI = np.zeros_like(gray_image)
        # Kích thước padding
        pad_size = (len(LoG_filter) - 1) // 2
        # Padding với kích thước pad_size và value = 0
        pad_width = ((pad_size, pad_size), (pad_size, pad_size))
        padded_image = np.pad(gray_image, pad_width, mode='constant', constant_values=0)
        
        # Gán giá trị vào ảnh LoG
        for x in range(pad_size, m + pad_size):
            for y in range(pad_size, n + pad_size):
                # Lấy các phần tử trong filter, phần tử đang xét nằm ở giữa
                neighbor = padded_image[x - pad_size:x + pad_size + 1, y - pad_size:y + pad_size + 1]
                # Tính giá trị của ảnh LoG
                LoG[x - pad_size, y - pad_size] = np.sum(LoG_filter * neighbor, axis=(0, 1))
        
        # Phát hiện biên cạnh
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                # # Tìm điểm zero-crossing
                neighbors = [LoG[i - 1, j], LoG[i + 1, j], LoG[i, j - 1], LoG[i, j + 1]]
                if any(np.sign(LoG[i, j]) != np.sign(neighbor) for neighbor in neighbors):
                    # Nếu điểm đó là zero crossing thì dùng cửa số 3x3 để tính local variance.
                    local_variance = np.var(gray_image[i - 1:i + 2, j - 1:j + 2])
                    # Nếu local variance > T thì đó là biên cạnh
                    if local_variance > T:
                        BEI[i, j] = 255
        
        return BEI
    except Exception as e:
        return str(e)
def non_maximum_suppression(gradient_magnitude, gradient_direction):
    height, width = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)
    for i in range(1, height-1):
        for j in range(1, width-1):
            angle = gradient_direction[i, j]
            q, r = 255, 255
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]
            elif 22.5 <= angle < 67.5: 
                q, r = gradient_magnitude[i+1, j-1], gradient_magnitude[i-1, j+1]
            elif 67.5 <= angle < 112.5:
                q, r = gradient_magnitude[i+1, j], gradient_magnitude[i-1, j]
            elif 112.5 <= angle < 157.5:
                q, r = gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]
            if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                suppressed[i, j] = gradient_magnitude[i, j]
            else:
                suppressed[i, j] = 0
    return suppressed

def gaussian_filter(filter_size, sigma):
    gaus_filter = np.zeros((filter_size,filter_size))
    bound = (filter_size-1)//2
    for x in range(-bound, bound+1):
        for y in range(-bound, bound+1):
            gaus_filter[x+bound][y+bound] = (1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))
    return gaus_filter

def gaussian_blur(image, filter_size, sigma):
    m, n = image.shape[:2]
    new_image = np.zeros_like(image)
    gaus_filter = gaussian_filter(filter_size,sigma)
    pad_size = (filter_size - 1)//2
    pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    for x in range(pad_size, m+pad_size):
        for y in range(pad_size, n+pad_size):
            neighbor = padded_image[x - pad_size:x + pad_size + 1, y - pad_size:y + pad_size + 1, :]
            new_image[x-pad_size,y-pad_size,:] = np.sum(gaus_filter[:,:,np.newaxis]*neighbor,axis = (0,1))
    return new_image.astype(int)

def canny_custom(image_data):
    try:
        _, image_data = image_data.split(';base64,')
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if gray_image is None:
            return None
        
        # Các tham số cho Canny
        filter_size = 5
        sigma = 1
        lower_threshold = 70
        upper_threshold = 20
        
        # Tạo gradient kernel
        kernel_x = (1/4)*np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
        kernel_y = (1/4)*np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        
        # Tính gradient theo kernel x và y
        gradient_x = cv2.filter2D(gray_image, -1, kernel_x)
        gradient_y = cv2.filter2D(gray_image, -1, kernel_y)
        
        # Áp dụng gaussian để làm trơn ảnh
        gray_image_blur = gaussian_blur(gray_image[:,:,np.newaxis], filter_size, sigma).squeeze()
        
        # Tính toán cạnh biên
        BEI = np.zeros_like(gray_image)
        suppressed = non_maximum_suppression(gray_image_blur, np.arctan2(gradient_y, gradient_x))
        strong_edges = suppressed >= upper_threshold
        weak_edges = (suppressed >= lower_threshold) & (suppressed < upper_threshold)
        height, width = weak_edges.shape
        for i in range(1, height-1):
            for j in range(1, width-1):
                if weak_edges[i,j] and strong_edges[i-1:i+2,j-1:j+2].any():
                    BEI[i,j] = 255
        BEI[strong_edges] = 255
        
        return BEI
    except Exception as e:
        return str(e)