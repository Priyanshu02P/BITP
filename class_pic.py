import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from .file_manager import select_file

class image:
    def __init__(self, img_mat):
        self.img_mat = img_mat

    @classmethod
    def from_file(cls, str_location):
        if not os.path.exists(str_location):
            raise FileNotFoundError(f"File does not exist: {str_location}")

        img_mat = cv.imread(str_location, cv.IMREAD_UNCHANGED)
        if img_mat is None:
            raise ValueError(f"Failed to load image. Invalid image file or path: {str_location}")
        return cls(img_mat)

    @classmethod
    def from_color(cls, RGB, size=(1080, 1080), alpha=255):
        r, g, b = RGB
        img_mat = np.full((size[1], size[0], 4), (b, g, r, alpha), dtype=np.uint8)
        return cls(img_mat)

    @classmethod
    def from_gradient(cls, RGB_one, RGB_two, visible_fraction=1.0, size=(1080, 1080)):
        height = size[1]
        width = size[0]

        bgr1 = tuple(reversed(RGB_one))
        bgr2 = tuple(reversed(RGB_two))

        gradient_start = int(height * (1 - visible_fraction))
        spectrum = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            if i < gradient_start:
                color = bgr1
            else:
                alpha = (i - gradient_start) / max(height - gradient_start - 1, 1)
                color = [
                    int((1 - alpha) * bgr1[c] + alpha * bgr2[c])
                    for c in range(3)
                ]
            spectrum[i, :] = color

        return cls(spectrum)
    
    @classmethod
    def from_select_file(cls):
        return cls.from_file(select_file())


    #object attributes

    def change(self):
        image = self.img_mat
        image_rgb = cv.cvtColor(image, cv.COLOR_BGRA2RGBA)
        image_bgr = cv.cvtColor(image, cv.COLOR_RGBA2BGRA)

        cv.imshow("1",image_rgb)
        cv.waitKey(0)
        cv.imshow("2",image_bgr)
        cv.waitKey(0)
        
        choice = input("Entre:")

        cv.destroyAllWindows()

        if(choice == 2):
            self.img_mat = image_bgr
        else:
            self.img_mat = image_rgb

    def show_img(self, str_window_name= "" ,wait = 0, size=(540,540)):
        img_mat = cv.resize(self.img_mat, size)
        cv.imshow(str_window_name, img_mat)
        cv.waitKey(wait)

    def resize(self, x, y):
        self.img_mat = cv.resize(self.img_mat, (x ,y))

    def show_x(self):
        return self.img_mat.shape[1]

    def show_y(self):
        return self.img_mat.shape[0] 
    
    def show_img_mat(self):
        return self.img_mat
    
    def set_img_mat(self, img_mat):
        self.img_mat = img_mat

    def make_rounded_edges(self, radius=0):
        image = Image.fromarray(self.img_mat)
        width, height = image.size
        rounded_image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([0, 0, width, height], radius, fill=255)
        rounded_image.paste(image, (0, 0), mask=mask)

        self.img_mat = np.array(rounded_image)

    def n_points_crop(self, *points):
        
        """
        Keeps only the region defined by any number of polygon points in the image,
        making all other pixels transparent.

        Parameters:
        - image (np.ndarray or PIL.Image): Input image
        - *points (tuple): Variable number of (x, y) coordinates defining the polygon

        Returns:
        - PIL.Image: Image with only the polygon region visible
        """

        if len(points) < 3:
            raise ValueError("At least 3 points are required to form a polygon.")

        
        self.img_mat = Image.fromarray(image)

        # Ensure image has alpha channel
        image = image.convert("RGBA")

        # Create transparent mask
        mask = Image.new("L", image.size, 0)

        # Draw polygon on the mask
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, fill=255)

        # Apply the mask
        result = Image.new("RGBA", image.size)
        result.paste(image, (0, 0), mask=mask)

        self.img_mat = np.array(result)
        
    def draw_oval(self, center=(540, 540), axes=(380, 350), color=(0, 255, 0), thickness=-1,angle = 360):
        cv.ellipse(self.img_mat, center, axes, 0, 0, angle, color, thickness)


    def draw_circle(self, center_coordinates, radius, color, thickness):
        self.img_mat = np.ascontiguousarray(self.img_mat, dtype=np.uint8)
        cv.circle(self.img_mat, center_coordinates, radius, color, thickness)

    def draw_line(self, start_point, end_point, color, thickness):
        self.img_mat = cv.line(self.img_mat, start_point, end_point, color, thickness)

    def draw_rectangle(self, start_point, end_point, color, thickness):
        self.img_mat =  cv.rectangle(self.img_mat, start_point, end_point, color, thickness)

    def mirror_image(self):
        self.img_mat = cv.flip(self.img_mat, 1)

    def rotate_image(self, angle=90):
        image = self.img_mat
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Rotation matrix
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform the rotation
        self.img_mat = cv.warpAffine(image, M, (w, h))
    
    def apply_top_transparency(self, visible_fraction=0.4):
        """
        Applies a vertical transparency gradient to an image: top is fully transparent,
        bottom is fully opaque starting from a defined visible_fraction of height.

        Args:
            image (np.ndarray): Input image in BGR or RGB format (H x W x 3).
            visible_fraction (float): Fraction (0–1) from bottom that is fully visible.
                                    The fade starts from height * (1 - visible_fraction).

        Returns:
            np.ndarray: Output image with BGRA channels (H x W x 4), where alpha varies vertically.

        """

        image = self.img_mat
        
        height, width = image.shape[:2]

        # Ensure 4 channels (add alpha if missing)
        if image.shape[2] == 3:
            image_bgra = np.dstack((image, np.ones((height, width), dtype=np.uint8) * 255))
        else:
            image_bgra = image.copy()

        # Calculate where the fade starts
        fade_start = int(height * (1 - visible_fraction))

        # Create an alpha gradient from top (0) to bottom (255)
        alpha = np.zeros((height,), dtype=np.uint8)
        if visible_fraction > 0:
            fade_length = height - fade_start
            gradient = np.linspace(0, 255, fade_length, endpoint=True, dtype=np.uint8)
            alpha[fade_start:] = gradient

        # Apply alpha gradient to all columns
        alpha_channel = np.tile(alpha[:, np.newaxis], (1, width))
        image_bgra[:, :, 3] = alpha_channel

        self.img_mat = image_bgra

    def apply_bottom_transparency(self, visible_fraction=0.4):
        """
        Applies a vertical transparency gradient to an image: top is fully transparent,
        bottom is fully opaque starting from a defined visible_fraction of height.

        Args:
            image (np.ndarray): Input image in BGR or RGB format (H x W x 3).
            visible_fraction (float): Fraction (0–1) from bottom that is fully visible.
                                    The fade starts from height * (1 - visible_fraction).

        Returns:
            np.ndarray: Output image with BGRA channels (H x W x 4), where alpha varies vertically.

        """

        image = self.img_mat
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Rotation matrix
        M = cv.getRotationMatrix2D(center, 180, 1.0)
        
        # Perform the rotation
        image = cv.warpAffine(image, M, (w, h))
        
        height, width = image.shape[:2]

        # Ensure 4 channels (add alpha if missing)
        if image.shape[2] == 3:
            image_bgra = np.dstack((image, np.ones((height, width), dtype=np.uint8) * 255))
        else:
            image_bgra = image.copy()

        # Calculate where the fade starts
        fade_start = int(height * (1 - visible_fraction))

        # Create an alpha gradient from top (0) to bottom (255)
        alpha = np.zeros((height,), dtype=np.uint8)
        if visible_fraction > 0:
            fade_length = height - fade_start
            gradient = np.linspace(0, 255, fade_length, endpoint=True, dtype=np.uint8)
            alpha[fade_start:] = gradient

        # Apply alpha gradient to all columns
        alpha_channel = np.tile(alpha[:, np.newaxis], (1, width))
        image_bgra[:, :, 3] = alpha_channel

        image = image_bgra

        # Rotation matrix
        M = cv.getRotationMatrix2D(center, 180, 1.0)
        
        # Perform the rotation
        image = cv.warpAffine(image, M, (w, h))

        self.img_mat = image

    def middle_x(self, x):
        """ Gives X for which image will be at the middle:"""
        image = self.img_mat
        img_x = image.shape[1] 
        return (x - (img_x)//2)

    def middle_y(self, y):
        """ Gives Y for which image will be at the middle:"""
        image = self.img_mat
        img_y = image.shape[0]
        return (y - (img_y)//2)

    def middle_x_y(self, x, y):
        """ Gives X and Y for which image will be at the middle:"""
        image = self.img_mat
        img_y = image.shape[0]
        img_x = image.shape[1]
        return (x - (img_x)//2), (y - (img_y)//2)

    def overlay_green_screen(self, Green_Screen, x=0, y=0):
        """
        Overlays a green screen image onto a background image at a given position.
        :param Green_Screen: object with show_img_mat() method returning the image
        :param x: x-coordinate to place the subject
        :param y: y-coordinate to place the subject
        """
        background = self.img_mat
        green_screen = Green_Screen.show_img_mat()

        # Remove alpha channel if present
        if green_screen.shape[2] == 4:
            green_screen = green_screen[:, :, :3]
        if background.shape[2] == 4:
            background = background[:, :, :3]

        # Convert to HSV and create mask for green
        hsv = cv.cvtColor(green_screen, cv.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        #mask = cv.inRange(hsv, lower_green, upper_green)

       # Create a mask to detect green pixels
        mask = cv.inRange(hsv, lower_green, upper_green)
        mask_inv = cv.bitwise_not(mask)

        # Extract the subject from the green screen image
        subject = cv.bitwise_and(green_screen, green_screen, mask=mask_inv)

        # Get dimensions of the subject and background
        sh, sw, _ = subject.shape  # Subject height & width           # Overlay position

        # Ensure the subject fits within the background dimensions
        bh, bw, _ = background.shape
        if x + sw > bw or y + sh > bh:
            raise ValueError("The subject goes out of the background boundaries!")

        # Extract region of interest (ROI) from background
        roi = background[y:y+sh, x:x+sw]

        # Extract the background part where the subject should be placed
        bg_part = cv.bitwise_and(roi, roi, mask=mask)

        # Combine subject with the extracted background part
        combined = cv.add(bg_part, subject)

        # Place the combined image onto the original background
        background[y:y+sh, x:x+sw] = combined

        self.img_mat = background

    def overlay_transparent(self, ForeGround, x=0, y=0):
        background = self.img_mat
        foreground = ForeGround.show_img_mat()

        # Ensure background has 4 channels
        if background.shape[2] == 3:
            bh, bw = background.shape[:2]
            alpha_bg = np.ones((bh, bw, 1), dtype=np.uint8) * 255
            background = np.concatenate((background, alpha_bg), axis=2)
            self.img_mat = background

        # Ensure foreground has 4 channels
        if foreground.shape[2] == 3:
            fh, fw = foreground.shape[:2]
            alpha_fg = np.ones((fh, fw, 1), dtype=np.uint8) * 255
            foreground = np.concatenate((foreground, alpha_fg), axis=2)

        # Now both have 4 channels, safe to proceed
        h, w = foreground.shape[:2]
        if x + w > background.shape[1] or y + h > background.shape[0]:
            print("⚠️ Foreground exceeds background dimensions!")
            return

        if (x < 0 or y < 0 or
        x + w > background.shape[1] or
        y + h > background.shape[0]):
            print("⚠️ Foreground exceeds background dimensions or invalid position!")
            return 

        roi = background[y:y+h, x:x+w]

        # Split alpha and color channels
        b, g, r, alpha = cv.split(foreground)
        foreground_rgb = cv.merge((b, g, r))
        alpha_mask = cv.cvtColor(alpha, cv.COLOR_GRAY2BGR).astype(float) / 255

        roi_rgb = roi[..., :3].astype(float)
        bg_part = (1 - alpha_mask) * roi_rgb
        fg_part = alpha_mask * foreground_rgb.astype(float)
        blended = bg_part + fg_part

        # Replace the color part in ROI
        background[y:y+h, x:x+w, :3] = blended.astype(np.uint8)
        background[y:y+h, x:x+w, 3] = 255  # Set alpha to fully opaque

        self.img_mat = background

    def overlay_image(self, Overlay, x=0, y=0, alpha=1.0):
        background = self.img_mat
        overlay = Overlay.show_img_mat()

        bh, bw = background.shape[:2]
        oh, ow = overlay.shape[:2]

        # Add alpha channel if not present
        if background.shape[2] == 3:
            alpha_channel = np.ones((bh, bw, 1), dtype=np.uint8) * 255
            background = np.concatenate((background, alpha_channel), axis=2)
            self.img_mat = background  # Update the stored image

        if x + ow > bw or y + oh > bh:
            print("⚠️ Overlay image goes out of bounds.")
            

        roi = background[y:y+oh, x:x+ow]
        roi_rgb = roi[..., :3].astype(float)
        overlay_rgb = overlay[..., :3].astype(float)

        blended = cv.addWeighted(roi_rgb, 1 - alpha, overlay_rgb, alpha, 0)
        background[y:y+oh, x:x+ow, :3] = blended.astype(np.uint8)
        background[y:y+oh, x:x+ow, 3] = 255  # Set alpha to fully opaque

        self.img_mat = background

    def apply_gausian_blur(self, ksize = (7,7), sigmaX = 0,  sigmaY=0):
        self.img_mat = cv.GaussianBlur(self.img_mat, ksize,sigmaX, sigmaY=sigmaY)

    def apply_zoom(self,zoom_factor=1.5):
        w = self.show_x()
        h = self.show_y()

        img = self.img_mat

        center_x, center_y = int(w // 2), int(h // 2)
        radius_x, radius_y = int(w // (2 * zoom_factor)), int(h // (2 * zoom_factor))

        cropped = img[
            center_y - radius_y:center_y + radius_y,
            center_x - radius_x:center_x + radius_x
        ]

        # Resize back to original size
        self.img_mat = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)

    def print_text(self, text,  x, y):
        pill = Image.fromarray(self.img_mat)

        draw = ImageDraw.Draw(pill)

        draw.text((x,y),text.get_text(),font=text.get_font(), fill=text.get_color())

        self.img_mat = np.array(pill)

    def save_to_location(self, str_location, filename):
        if cv.imwrite(os.path.join(str_location, filename), self.img_mat):
            print(filename + " Saved at " + str_location)
        else:
            print("Coudn't save file")

    def print_para(self, para: list, txt: "text", text_box : tuple , center_x:int ,center_y: int, linespace: float=0.6, change: bool = True , rate:int = 1 , padding = 0):
        txt.set_text("0")
        height = txt.text_height()*linespace*(len(para) -1)
        width = 0

        top = text_box[0]
        bottom = text_box[2]
        left = text_box[1]
        right = text_box[3]

        text_max_h = bottom - top - padding*2
        text_max_w = right - left - padding * 2

        def calculate_height():
            nonlocal height, width 
            txt.set_text("0")
            height = txt.text_height()*linespace*(len(para) -1)
            width = 0
            for p in para:
                txt.set_text(p)
                height += txt.text_height()
                if(txt.text_width() > width):
                    width = txt.text_width()

        calculate_height()

        if(change):
            while(height > text_max_h):
                txt.set_size(txt.get_size() - rate)

                calculate_height()

            while(width > text_max_w):
                txt.set_size(txt.get_size() - rate)

                calculate_height()

        y = center_y - (height//2) - top

        for i in range(len(para)):
            if(i != 0):
                y += txt.text_height()
                txt.set_text("0")
                y += linespace*txt.text_height()
            txt.set_text(para[i])
            self.print_text(txt, txt.center_text_X(center_x) , y)


            


    

class text:
    def __init__(self, text, font_location, size = 30, str_color = "white"):
        self.text = text
        self.location = font_location
        self.size = size
        self.font = ImageFont.truetype(font_location, size)
        self.color = str_color

    #object attributes
    def set_font(self, font_location, size = 30):
        self.size = size
        self.location = font_location
        self.font = ImageFont.truetype(font_location, size)

    def set_size(self, size):
        self.size = size
        self.font = ImageFont.truetype(self.location, size)

    def set_text(self, text):
        self.text = text

    def set_color(self, str_color):
        self.color = str_color

    def get_font(self):
        return self.font
    
    def get_size(self):
        return self.size 

    def get_text(self):
        return self.text 

    def get_color(self):
        return self.color

    def print_text(self, image, x, y):
        pill = Image.fromarray(image.show_img_mat())

        draw = ImageDraw.Draw(pill)

        draw.text((x,y),self.text,font=self.font, fill=self.color)

        image.set_img_mat(np.array(pill))

        return image
   

    def getFontSize(self,width, height): 
        location = self.location
        size = self.size
        font = self.font
        text = self.text
        left, top, right, bottom = font.getbbox(text)  # Get text bounding box
        text_width = right - left
        text_height = bottom - top
        while(text_width>width):
            size -= 1
            font = ImageFont.truetype(location,size)
            left, top, right, bottom = font.getbbox(text)  # Get text bounding box
            text_width = right - left

        while(text_height>height):
            size -= 1
            font = ImageFont.truetype(location,size)
            left, top, right, bottom = font.getbbox(text)  # Get text bounding box
            text_height = bottom - top

        self.font = font

    def bottom_x_y(self, x, y):
        text = self.text
        font = self.font

        left, top, right, bottom = font.getbbox(text)  # Get text bounding box
        text_width = right - left
        text_height = bottom - top

        return x - text_width, y - text_height
    
    def bottom_x(self , x):
        text = self.text
        font = self.font

        left, top, right, bottom = font.getbbox(text)  # Get text bounding box
        text_width = right - left

        return x - text_width

    def bottom_y(self , y):
        text = self.text
        font = self.font

        left, top, right, bottom = font.getbbox(text)  # Get text bounding box
        text_height = bottom - top

        return y - text_height

    def text_height(self):
        text = self.text
        font = self.font

        left, top, right, bottom = font.getbbox(text)  # Get text bounding box
        return bottom - top
    
    def text_width(self):
        text = self.text
        font = self.font

        left, top, right, bottom = font.getbbox(text)  # Get text bounding box
        return right - left

    def center_text_x_y(self, desired_x_center,desired_y_centre):
        """
        Centers text horizontally at a specific Y position.

        Parameters:
        - draw (ImageDraw): PIL ImageDraw object.
        - text (str): The text to be centered.
        - font (ImageFont): The font object.
        - y_pos (int): The Y-coordinate where text should be placed.
        - desired_x_center (int): The X-coordinate for centering the text.

        Returns:
        - (int, int): The (x, y) coordinates where the text should start.
        """
        text = self.text
        font = self.font

        left, top, right, bottom = font.getbbox(text)  # Get text bounding box
        text_width = right - left
        text_height = bottom - top
        x_pos = desired_x_center - (text_width // 2 ) # Center horizontally
        y_pos = desired_y_centre - (text_height//2) - top

        return x_pos,y_pos   

    def center_text_X(self, desired_x_center):
        """
        Centers text horizontally at a specific X position.

        Parameters:
        - draw (ImageDraw): PIL ImageDraw object.
        - text (str): The text to be centered.
        - font (ImageFont): The font object.
        - y_pos (int): The Y-coordinate where text should be placed.
        - desired_x_center (int): The X-coordinate for centering the text.

        Returns:
        - (int, int): The (x, y) coordinates where the text should start.
        """

        font = self.font
        text = self.text
        left, top, right, bottom = font.getbbox(text)  # Get text bounding box
        text_width = right - left
        x_pos = desired_x_center - (text_width // 2)  # Center horizontally
        
        return x_pos
    
    def center_text_Y(self, desired_y_center):
        """
        Centers text horizontally at a specific X position.

        Parameters:
        - draw (ImageDraw): PIL ImageDraw object.
        - text (str): The text to be centered.
        - font (ImageFont): The font object.
        - y_pos (int): The Y-coordinate where text should be placed.
        - desired_x_center (int): The X-coordinate for centering the text.

        Returns:
        - (int, int): The (x, y) coordinates where the text should start.
        """
        text = self.text
        font = self.font

        left, top, right, bottom = font.getbbox(text)  # Get text bounding box
        text_height = bottom - top
        y_pos = desired_y_center - (text_height // 2 ) - top # Center horizontally
        
        return y_pos


    