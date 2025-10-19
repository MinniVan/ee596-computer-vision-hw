import cv2
import numpy as np
import os


class ComputerVisionAssignment:
    def __init__(self, image_path, binary_image_path):
        self.image = cv2.imread(image_path)
        self.binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    def check_package_versions(self):
        # Ungraded
        import numpy as np
        import matplotlib
        import cv2
        '''
        print(np.__version__)
        print(matplotlib.__version__)
        print(cv2.__version__)
        '''

    def load_and_analyze_image(self):
        """
        Fill your code here
        """
        image = self.image
        Pixel_data_type = image.dtype

        height, width, channels = image.shape
        Image_shape = (height, width, channels)
        Image_data_type_dict = {1: "Grayscale", 3: "BGR"}
        Image_data_type = Image_data_type_dict[channels]

        #for i in range(3):
        #    print('-')
        #    print(image[i][:5])
        '''
        print("----- Task 1: Load and analyze the image ----")

        print(f"Image data type: {Image_data_type}")
        print(f"Pixel data type: {Pixel_data_type}")
        print(f"Image dimensions: {Image_shape}")

        print("----- Task 1: Load and analyze the image ----")
        '''
        return Image_data_type, Pixel_data_type, Image_shape
    
    def create_red_image(self):
        """
        Fill your code here
        """
        red_image = self.image.copy()
        red_image[:, :, 0] = 0 # Blue
        red_image[:, :, 1] = 0 # Green
        '''
        plt.figure()
        plt.imshow(red_image[:,:,::-1])
        plt.title("Task 2: Create a red image")
        '''
        return red_image


    def create_photographic_negative(self):
        """
        Fill your code here
        """
        image = self.image
        negative_image = image.copy()
        negative_image = 255 - negative_image
        '''
        plt.figure()
        plt.imshow(negative_image[:,:,::-1])
        plt.title("Task 3: Create a photographic negative")
        '''
        return negative_image
    
    def swap_color_channels(self):
        swapped_image = self.image.copy()
        # swap B (0) and R (2)
        swapped_image[:, :, 0], swapped_image[:, :, 2] = swapped_image[:, :, 2].copy(), swapped_image[:, :, 0].copy()
        '''
        plt.figure()
        plt.imshow(swapped_image[:,:,::-1])
        plt.title("Task 4: Swap color channels")
        '''
        return swapped_image

    def foliage_detection(self):
        b = self.image[:, :, 0]
        g = self.image[:, :, 1]
        r = self.image[:, :, 2]

        mask = (g >= 50) & (b < 50) & (r < 50)
        foliage_image = np.where(mask, 255, 0).astype(np.uint8)
        '''
        plt.figure()
        plt.imshow(foliage_image, cmap="gray")
        plt.title("Task 5: Foliage detection")
        '''
        return foliage_image
    
    def shift_image(self):
        """
        Fill your code here
        """
        rows, cols = self.image.shape[:2]

        # transfomation matrx from leture [ 1, 0, tx], [0, 1, ty] ]

        M = np.float32([[1, 0, 200], [0, 1, 100]])

        im = self.image.copy()
        shifted_image = cv2.warpAffine(im, M, (cols, rows))
        '''
        plt.figure()
        plt.imshow(shifted_image[:,:,::-1])
        plt.title("Task 6: Image shifted right 200 pixels and down 100 pixels")
        '''
        return shifted_image
    
    def rotate_image(self):
        """
        Fill your code here
        """
        rotated_image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        '''
        plt.figure()
        plt.imshow(rotated_image[:,:,::-1])
        plt.title("Task 7: Rotate image CW by 90 degrees")
        '''
        return rotated_image


    def similarity_transform(self, scale, theta, shift):
        """
        Fill your code here
        """
        img = self.image

        height, width = img.shape[:2]

        # scale -> rotate (theta CCW) --> translate 
        rad = np.deg2rad(theta)
        cos_ = np.cos(rad) * float(scale)
        sin_ = np.sin(rad) * float(scale)

        A = np.array([[cos_, -sin_],
                      [sin_, cos_]], dtype=np.float64)
        
        t = np.array([float(shift[0]), float(shift[1])], dtype=np.float64)

        M_forward = np.concatenate([A, t.reshape(2,1)], axis=1).astype(np.float32)

        # invere mapping

        M_inv = cv2.invertAffineTransform(M_forward)
        transformed_image = cv2.warpAffine(
            img, M_inv, (width, height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        '''
        plt.figure()
        plt.imshow(transformed_image[:,:,::-1])
        plt.title("Task 8: Similarity transform")
        '''
        return transformed_image
    
    def convert_to_grayscale(self):
        """
        Fill your code here
        """
        img = self.image.astype(np.float64)
        b_channel = img[:, :, 0]
        g_channel = img[:, :, 1]
        r_channel = img[:, :, 2]

        gray_image = np.rint((3*r_channel + 6*g_channel + 1*b_channel) / 10.0)

        gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)
        '''
        plt.figure()
        plt.imshow(gray_image, cmap="gray")
        plt.title("Task 9: Grayscale conversion")
        '''
        return gray_image
    
    def compute_moments(self):
        """
        Fill your code here

        """

        # binary mask (0 1)
        I = (self.binary_image).astype(np.float64)

        height, width = I.shape

        y_coords = np.arange(height, dtype=np.float64)[:, None] # rows
        x_coords = np.arange(width, dtype=np.float64)[None, :] # cols

        # raw moments
        m00 = I.sum()
        m10 = (x_coords * I).sum()
        m01 = (y_coords * I).sum()


        # centriods
        x_bar = m10 / m00
        y_bar = m01 / m00

        # central 2nd order moments
        mu20 = ((x_coords - x_bar)**2 * I).sum()
        mu02 = ((y_coords - y_bar)**2 * I).sum()
        mu11 = (((x_coords - x_bar) * (y_coords - y_bar)) * I).sum()
        '''
        # Print the results
        print()
        print("Task 10: Moments of a binary image")
        print("First-Order Moments:")
        print(f"Standard (Raw) Moments: M00 = {m00}, M10 = {m10}, M01 = {m01}")
        print("Centralized Moments:")
        print(f"x_bar = {x_bar}, y_bar = {y_bar}")
        print("Second-Order Centralized Moments:")
        print(f"mu20 = {mu20}, mu02 = {mu02}, mu11 = {mu11}")
        '''

        return m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11


    def compute_orientation_and_eccentricity(self):
        """
        Fill your code here
        """
        m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11 = self.compute_moments()


        if m00 == 0:
            glasses_with_ellipse = self.image.copy()
            orientation = 0.0
            eccentricity = 0.0
            return orientation, eccentricity, glasses_with_ellipse

        # orientation
        theta_rad_ccw = 0.5 * np.arctan2(2.0 * mu11, (mu20 - mu02))
        theta_deg_ccw = np.degrees(theta_rad_ccw) % 180.0
        orientation = theta_deg_ccw

        # covariance matrix
        covariance_matrix = np.array([
            [mu20 / m00, mu11 / m00],
            [mu11 / m00, mu02 / m00]
        ], dtype=np.float64)

        eigvals, _ = np.linalg.eigh(covariance_matrix)
        minor_var, major_var = float(eigvals[0]), float(eigvals[1])

        semi_major = 2.0 * np.sqrt(max(major_var, 0.0))
        semi_minor = 2.0 * np.sqrt(max(minor_var, 0.0))

        # ecentricity
        if semi_major <= 1e-12:
            eccentricity = 0.0
        else:
            eccentricity = float(np.sqrt(1.0 - (semi_minor**2) / (semi_major**2 + 1e-12)))
        
        # dwar ellipse in red
        glasses_with_ellipse = self.image.copy()  # BGR image
        center = (int(round(x_bar)), int(round(y_bar)))
        axes   = (max(1, int(round(semi_major))), max(1, int(round(semi_minor))))  # avoid zero axes
        cv2.ellipse(glasses_with_ellipse, center, axes,
                    theta_deg_ccw, 0, 360, (0, 0, 255), 1)
        '''
        plt.figure()
        plt.imshow(cv2.cvtColor(glasses_with_ellipse, cv2.COLOR_BGR2RGB))
        plt.title("Task 11: Orientation and eccentricity of a binary image")
        plt.show()
        '''
        return orientation, eccentricity, glasses_with_ellipse

if __name__ == "__main__":

    assignment = ComputerVisionAssignment("picket_fence.png", "binary_image.png")

    # Task 0: Check package versions
    assignment.check_package_versions()

    # Task 1: Load and analyze the image
    assignment.load_and_analyze_image()

    # Task 2: Create a red image
    red_image = assignment.create_red_image()

    # Task 3: Create a photographic negative
    negative_image = assignment.create_photographic_negative()

    # Task 4: Swap color channels
    swapped_image = assignment.swap_color_channels()

    # Task 5: Foliage detection
    foliage_image = assignment.foliage_detection()

    # Task 6: Shift the image
    shifted_image = assignment.shift_image()

    # Task 7: Rotate the image
    rotated_image = assignment.rotate_image()

    # Task 8: Similarity transform
    transformed_image = assignment.similarity_transform(
        scale=2.0, theta=45.0, shift=[100, 100]
    )

    # Task 9: Grayscale conversion
    gray_image = assignment.convert_to_grayscale()

    glasses_assignment = ComputerVisionAssignment(
        "glasses_outline.png", "glasses_outline.png"
    )

    # Task 10: Moments of a binary image
    glasses_assignment.compute_moments()

    # Task 11: Orientation and eccentricity of a binary image
    orientation, eccentricity, glasses_with_ellipse = (
        glasses_assignment.compute_orientation_and_eccentricity()
    )
