import torch
import torch.nn as nn

class RGB2LAB(nn.Module):
    
    def __init__(self):
        super(RGB2LAB, self).__init__()
        self.rgb_to_xyz = nn.Parameter(torch.tensor(
                                        [  # X        Y          Z
                                            [0.412453, 0.212671, 0.019334],  # R
                                            [0.357580, 0.715160, 0.119193],  # G
                                            [0.180423, 0.072169, 0.950227],  # B
                                        ], dtype=torch.float), requires_grad=False)
        self.fxfyfz_to_lab = nn.Parameter(torch.tensor(
                                            [
                                                [0.0,  500.0,    0.0],  # fx
                                                [116.0, -500.0,  200.0],  # fy
                                                [0.0,    0.0, -200.0],  # fz
                                            ], dtype=torch.float), requires_grad=False)
        self.xyz_to_rgb_mult = nn.Parameter(torch.tensor([0.950456, 1.0, 1.088754], dtype=torch.float).reshape(1, 3, 1, 1), requires_grad=False)
        self.lab_to_fxfyfz_offset = nn.Parameter(torch.tensor([16.0, 0.0, 0.0], dtype=torch.float).reshape(1, 3, 1, 1), requires_grad=False)
        
    def forward(self, img):
        """ PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: image to be adjusted
        :returns: adjusted image
        :rtype: Tensor

        """
        img = img.contiguous()

        img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
                                                                       min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(0.04045).float()
        img = torch.einsum('bcxy,ck->bkxy', img, self.rgb_to_xyz)
        img = torch.mul(img, 1/self.xyz_to_rgb_mult)

        epsilon = 6/29

        img = ((img / (3.0 * epsilon**2) + 4.0/29.0) * img.le(epsilon**3).float()) + \
            (torch.clamp(img, min=0.0001) **
             (1.0/3.0) * img.gt(epsilon**3).float())

        img = torch.einsum('bcxy,ck->bkxy', img, self.fxfyfz_to_lab) - self.lab_to_fxfyfz_offset

        '''
        L_chan: black and white with input range [0, 100]
        a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
        [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
        '''
        img[:, 0, :, :] = img[:, 0, :, :]/100
        img[:, 1, :, :] = (img[:, 1, :, :]/110 + 1)/2
        img[:, 2, :, :] = (img[:, 2, :, :]/110 + 1)/2

        img = img.contiguous()
        return img

    
class LAB2RGB(nn.Module):
    
    def __init__(self):
        super(LAB2RGB, self).__init__()
        self.xyz_to_rgb = nn.Parameter(torch.tensor(
                                        [   # X          Y           Z
                                            [3.2404542, -0.9692660,  0.0556434],  # R
                                            [-1.5371385,  1.8760108, -0.2040259],  # G
                                            [-0.4985314,  0.0415560,  1.0572252],  # B
                                        ], dtype=torch.float), requires_grad=False)
        self.lab_to_fxfyfz = nn.Parameter(torch.tensor(
                                            [   # X       Y         Z
                                                [1/116.0, 1/116.0, 1/116.0],  # R
                                                [1/500.0, 0, 0],  # G
                                                [0, 0, -1/200.0],  # B
                                            ], dtype=torch.float), requires_grad=False)
        self.xyz_to_rgb_mult = nn.Parameter(torch.tensor([0.950456, 1.0, 1.088754], dtype=torch.float).reshape(1, 3, 1, 1), requires_grad=False)
        self.lab_to_fxfyfz_offset = nn.Parameter(torch.tensor([16.0, 0.0, 0.0], dtype=torch.float).reshape(1, 3, 1, 1), requires_grad=False)

    def forward(self, img):
        """ PyTorch implementation of LAB to RGB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: image to be adjusted
        :returns: adjusted image
        :rtype: Tensor
        """                
        img = img.contiguous()

        channel0 = torch.unsqueeze(img[:, 0, :, :] * 100, dim=1)
        channel1 = torch.unsqueeze(((img[:, 1, :, :] * 2)-1)*110, dim=1)
        channel2 = torch.unsqueeze(((img[:, 2, :, :] * 2)-1)*110, dim=1)

        img = torch.cat([channel0, channel1, channel2], dim=1)

        img = torch.einsum('bcxy,ck->bkxy', 
                           img + self.lab_to_fxfyfz_offset, 
                           self.lab_to_fxfyfz)

        epsilon = 6.0/29.0

        img = (((3.0 * epsilon**2 * (img-4.0/29.0)) * img.le(epsilon).float()) +
               ((torch.clamp(img, min=0.0001)**3.0) * img.gt(epsilon).float()))

        # denormalize for D65 white point
        img = torch.mul(img, self.xyz_to_rgb_mult)

        img = torch.einsum('bcxy,ck->bkxy', img, self.xyz_to_rgb)
        img = (img * 12.92 * img.le(0.0031308).float()) + ((torch.clamp(img,
                                                                        min=0.0001) ** (1/2.4) * 1.055) - 0.055) * img.gt(0.0031308).float()

        img = img.contiguous()

        return img
    

class HSV2RGB(nn.Module):

    def __init__(self):
        super(HSV2RGB, self).__init__()
    
    def forward(self, img):
        """Converts a HSV image to RGB
        PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

        :param img: HSV image
        :returns: RGB image
        :rtype: Tensor

        """
        img=torch.clamp(img, 0.0, 1.0)

        m1 = 0
        m2 = (img[:, 2, :, :]*(1-img[:, 1, :, :])-img[:, 2, :, :])/60
        m3 = 0
        m4 = -1*m2
        m5 = 0

        r = img[:, 2, :, :]+torch.clamp(img[:, 0, :, :]*360-0, 0.0, 60.0)*m1+torch.clamp(img[:, 0, :, :]*360-60, 0.0, 60.0)*m2+torch.clamp(
            img[:, 0, :, :]*360-120, 0.0, 120.0)*m3+torch.clamp(img[:, 0, :, :]*360-240, 0.0, 60.0)*m4+torch.clamp(img[:, 0, :, :]*360-300, 0.0, 60.0)*m5
        del m1, m2, m3, m4, m5

        m1 = (img[:, 2, :, :]-img[:, 2, :, :]*(1-img[:, 1, :, :]))/60
        m2 = 0
        m3 = -1*m1
        m4 = 0

        g = img[:, 2, :, :]*(1-img[:, 1, :, :])+torch.clamp(img[:, 0, :, :]*360-0, 0.0, 60.0)*m1+torch.clamp(img[:, 0, :, :]*360-60,
            0.0, 120.0)*m2+torch.clamp(img[:, 0, :, :]*360-180, 0.0, 60.0)*m3+torch.clamp(img[:, 0, :, :]*360-240, 0.0, 120.0)*m4
        del m1, m2, m3, m4

        m1 = 0
        m2 = (img[:, 2, :, :]-img[:, 2, :, :]*(1-img[:, 1, :, :]))/60
        m3 = 0
        m4 = -1*m2

        b = img[:, 2, :, :]*(1-img[:, 1, :, :])+torch.clamp(img[:, 0, :, :]*360-0, 0.0, 120.0)*m1+torch.clamp(img[:, 0, :, :]*360 -
            120, 0.0, 60.0)*m2+torch.clamp(img[:, 0, :, :]*360-180, 0.0, 120.0)*m3+torch.clamp(img[:, 0, :, :]*360-300, 0.0, 60.0)*m4
        del m1, m2, m3, m4

        img = torch.stack((r, g, b), 1)
        del r, g, b

        img = img.contiguous()
        img = torch.clamp(img, 0.0, 1.0)

        return img


class RGB2HSV(nn.Module):

    def __init__(self):
        super(RGB2HSV, self).__init__()
        self.comparison_zero = nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=False)

    @staticmethod
    def non_nan_inv(tensor):
        # Hacky way of not creating new tensor from scratch
        # and run into device issues.
        tensor_inv = 0.0 * tensor
        tensor_inv[tensor != 0] = 1/(tensor[tensor != 0])

        return tensor_inv

    def forward(self, img):
        """Converts an RGB image to HSV
        PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

        :param img: RGB image
        :returns: HSV image
        :rtype: Tensor

        """
        img=torch.clamp(img, 10**(-9), 1.0)       

        # Shape (b, c, x, y)
        img = img.contiguous()

        # Shape (b, x, y)
        mx = torch.max(img, 1)[0]
        mn = torch.min(img, 1)[0]

        df = torch.add(mx, -1.0 * mn)

        # Each channel is shape (b, x, y) tensor
        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        # New channel 0, hue (see: https://www.rapidtables.com/convert/color/rgb-to-hsv.html)
        df_inv = RGB2HSV.non_nan_inv(df)        
        img[:, 0, :, :] = torch.where(df == 0.0, 
                                      self.comparison_zero,
                                      ((g-b)*df_inv)*r.eq(mx).float() + (2.0+(b-r)*df_inv)
                                      * g.eq(mx).float() + (4.0+(r-g)*df_inv)*b.eq(mx).float())
        img[:, 0, :, :] = img[:, 0, :, :]*60.0

        # Convert hue to range 0 to 360
        img[:, 0, :, :] = img[:, 0, :, :].lt(0.0).float(
        )*(img[:, 0, :, :]+360) + img[:, 0, :, :].ge(0.0).float()*(img[:, 0, :, :])

        img[:, 0, :, :] = img[:, 0, :, :]/360

        # Set saturation and value, remaining channels
        mx_inv = RGB2HSV.non_nan_inv(mx)
        img[:, 1, :, :] = torch.where(mx == 0.0,
                                      self.comparison_zero,
                                      mx.ne(0.0).float()*(df*mx_inv) + mx.eq(0.0).float()*(0.0))
        img[:, 2, :, :] = mx

        img = torch.clamp(img, 10**(-9), 1.0)

        return img
